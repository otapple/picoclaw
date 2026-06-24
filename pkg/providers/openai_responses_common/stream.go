package openai_responses_common

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3/responses"

	"github.com/sipeed/picoclaw/pkg/providers/common"
	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

// StreamingRequest contains the transport metadata needed to call the
// Responses API without delegating SSE parsing to the OpenAI SDK.
type StreamingRequest struct {
	HTTPClient *http.Client
	APIBase    string
	APIKey     string
	UserAgent  string
	Headers    map[string]string
	Params     responses.ResponseNewParams
}

// StreamResponseMeta is attached to parser errors so provider logs can include
// useful response context without dumping full payloads.
type StreamResponseMeta struct {
	StatusCode  int
	ContentType string
	RequestID   string
}

// ResponsesStreamError wraps HTTP and SSE parsing failures with bounded
// diagnostic context.
type ResponsesStreamError struct {
	StatusCode  int
	ContentType string
	RequestID   string
	EventType   string
	Preview     string
	Err         error
}

func (e *ResponsesStreamError) Error() string {
	if e == nil {
		return "<nil>"
	}

	parts := make([]string, 0, 5)
	if e.StatusCode != 0 {
		parts = append(parts, fmt.Sprintf("status=%d", e.StatusCode))
	}
	if e.ContentType != "" {
		parts = append(parts, fmt.Sprintf("content_type=%q", e.ContentType))
	}
	if e.RequestID != "" {
		parts = append(parts, fmt.Sprintf("request_id=%q", e.RequestID))
	}
	if e.EventType != "" {
		parts = append(parts, fmt.Sprintf("event=%q", e.EventType))
	}
	if e.Preview != "" {
		parts = append(parts, fmt.Sprintf("preview=%q", e.Preview))
	}

	msg := "responses stream error"
	if len(parts) > 0 {
		msg += " (" + strings.Join(parts, ", ") + ")"
	}
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *ResponsesStreamError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// DoStreamingResponseRequest posts a streaming Responses API request and parses
// the SSE response while tolerating legal heartbeat/comment frames.
func DoStreamingResponseRequest(ctx context.Context, req StreamingRequest) (*protocoltypes.LLMResponse, error) {
	apiBase := strings.TrimRight(strings.TrimSpace(req.APIBase), "/")
	if apiBase == "" {
		return nil, errors.New("api_base is required for responses stream request")
	}

	body, err := marshalStreamingParams(req.Params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal responses request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, apiBase+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create responses request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if strings.TrimSpace(req.UserAgent) != "" {
		httpReq.Header.Set("User-Agent", req.UserAgent)
	}
	if strings.TrimSpace(req.APIKey) != "" {
		httpReq.Header.Set("Authorization", "Bearer "+req.APIKey)
	}
	for key, value := range req.Headers {
		if strings.TrimSpace(key) == "" {
			continue
		}
		httpReq.Header.Set(key, value)
	}

	client := req.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send responses request: %w", err)
	}
	defer resp.Body.Close()

	meta := StreamResponseMeta{
		StatusCode:  resp.StatusCode,
		ContentType: resp.Header.Get("Content-Type"),
		RequestID:   responseRequestID(resp.Header),
	}
	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 512))
		if readErr != nil {
			return nil, fmt.Errorf("failed to read responses error body: %w", readErr)
		}
		return nil, &ResponsesStreamError{
			StatusCode:  meta.StatusCode,
			ContentType: meta.ContentType,
			RequestID:   meta.RequestID,
			Preview:     common.ResponsePreview(body, 256),
			Err:         fmt.Errorf("HTTP %d from Responses API", resp.StatusCode),
		}
	}

	return ParseStreamingResponse(ctx, resp.Body, meta)
}

// ParseStreamingResponse parses Responses API SSE data into the provider-level
// LLMResponse. Empty data events and comment-only heartbeat frames are ignored.
func ParseStreamingResponse(
	ctx context.Context,
	reader io.Reader,
	meta StreamResponseMeta,
) (*protocoltypes.LLMResponse, error) {
	var resp *responses.Response
	var streamedText strings.Builder
	streamedOutputItems := make([]responses.ResponseOutputItemUnion, 0)

	processEvent := func(eventType, data string) (bool, error) {
		trimmed := strings.TrimSpace(data)
		if trimmed == "" {
			return false, nil
		}
		if trimmed == "[DONE]" {
			return true, nil
		}

		var envelope struct {
			Error json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal([]byte(trimmed), &envelope); err == nil && len(envelope.Error) > 0 &&
			string(envelope.Error) != "null" {
			return false, streamError(meta, eventType, trimmed, errors.New("received error while streaming"))
		}

		var evt responses.ResponseStreamEventUnion
		if err := json.Unmarshal([]byte(trimmed), &evt); err != nil {
			return false, streamError(
				meta,
				eventType,
				trimmed,
				fmt.Errorf("failed to decode responses stream event: %w", err),
			)
		}

		switch evt.Type {
		case "response.output_text.delta":
			streamedText.WriteString(evt.Delta)
		case "response.output_item.done":
			itemEvt := evt.AsResponseOutputItemDone()
			if itemEvt.Item.Type != "" {
				streamedOutputItems = append(streamedOutputItems, itemEvt.Item)
			}
		case "response.completed", "response.failed", "response.incomplete":
			evtResp := evt.Response
			if evtResp.ID != "" {
				evtRespCopy := evtResp
				resp = &evtRespCopy
			}
		case "error":
			return false, streamError(meta, eventName(eventType, evt.Type), trimmed, errors.New("received error event"))
		}

		return false, nil
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	var eventType string
	var data strings.Builder
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			done, err := processEvent(eventType, data.String())
			if err != nil {
				return nil, err
			}
			if done {
				break
			}
			eventType = ""
			data.Reset()
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}

		name, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		value = strings.TrimPrefix(value, " ")
		switch name {
		case "event":
			eventType = value
		case "data":
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(value)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, streamError(meta, eventType, "", fmt.Errorf("streaming read error: %w", err))
	}
	if data.Len() > 0 {
		_, err := processEvent(eventType, data.String())
		if err != nil {
			return nil, err
		}
	}

	if resp == nil {
		return nil, errors.New("responses stream ended without completed response")
	}
	if len(resp.Output) == 0 && len(streamedOutputItems) > 0 {
		resp.Output = streamedOutputItems
	}

	parsed := ParseResponseFromStruct(resp)
	if parsed.Content == "" && streamedText.Len() > 0 {
		parsed.Content = streamedText.String()
	}
	return parsed, nil
}

func marshalStreamingParams(params responses.ResponseNewParams) ([]byte, error) {
	data, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}

	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		return nil, err
	}
	body["stream"] = true
	return json.Marshal(body)
}

func streamError(meta StreamResponseMeta, eventType, preview string, err error) *ResponsesStreamError {
	return &ResponsesStreamError{
		StatusCode:  meta.StatusCode,
		ContentType: meta.ContentType,
		RequestID:   meta.RequestID,
		EventType:   eventType,
		Preview:     common.ResponsePreview([]byte(preview), 256),
		Err:         err,
	}
}

func eventName(sseType, jsonType string) string {
	if strings.TrimSpace(sseType) != "" {
		return sseType
	}
	return jsonType
}

func responseRequestID(headers http.Header) string {
	for _, key := range []string{"x-request-id", "openai-request-id", "request-id"} {
		if value := headers.Get(key); value != "" {
			return value
		}
	}
	return ""
}
