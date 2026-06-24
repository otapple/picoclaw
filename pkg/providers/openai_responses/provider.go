package openai_responses

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers/common"
	orc "github.com/sipeed/picoclaw/pkg/providers/openai_responses_common"
	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

type (
	LLMResponse            = protocoltypes.LLMResponse
	Message                = protocoltypes.Message
	StreamChunk            = protocoltypes.StreamChunk
	ToolDefinition         = protocoltypes.ToolDefinition
	ToolFunctionDefinition = protocoltypes.ToolFunctionDefinition
)

const defaultAPIBase = "https://api.openai.com/v1"

// Provider implements the OpenAI Responses API with API-key authentication.
type Provider struct {
	httpClient    *http.Client
	apiKey        string
	apiBase       string
	userAgent     string
	customHeaders map[string]string
}

// NewProvider creates an OpenAI Responses API provider.
func NewProvider(
	apiKey string,
	apiBase string,
	proxy string,
	userAgent string,
	requestTimeoutSeconds int,
	customHeaders map[string]string,
) *Provider {
	apiBase = strings.TrimRight(strings.TrimSpace(apiBase), "/")
	if apiBase == "" {
		apiBase = defaultAPIBase
	}

	httpClient := common.NewHTTPClient(proxy)
	if requestTimeoutSeconds > 0 {
		httpClient.Timeout = time.Duration(requestTimeoutSeconds) * time.Second
	}

	return &Provider{
		httpClient:    httpClient,
		apiKey:        apiKey,
		apiBase:       apiBase,
		userAgent:     userAgent,
		customHeaders: cloneHeaders(customHeaders),
	}
}

func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	if p == nil || p.httpClient == nil {
		return nil, fmt.Errorf("openai responses provider is not configured")
	}
	if strings.TrimSpace(p.apiKey) == "" {
		return nil, fmt.Errorf("api_key is required for openai-responses provider")
	}

	params := buildParams(messages, tools, model, options)
	parsed, err := orc.DoStreamingResponseRequest(ctx, orc.StreamingRequest{
		HTTPClient: p.httpClient,
		APIBase:    p.apiBase,
		APIKey:     p.apiKey,
		UserAgent:  p.userAgent,
		Headers:    p.customHeaders,
		Params:     params,
	})
	if err != nil {
		fields := map[string]any{
			"model":          model,
			"messages_count": len(messages),
			"tools_count":    len(tools),
			"api_base":       p.apiBase,
			"error":          err.Error(),
		}
		var apiErr *openai.Error
		if errors.As(err, &apiErr) {
			fields["status_code"] = apiErr.StatusCode
			fields["api_type"] = apiErr.Type
			fields["api_code"] = apiErr.Code
			fields["api_param"] = apiErr.Param
			fields["api_message"] = apiErr.Message
			if apiErr.Response != nil {
				fields["request_id"] = apiErr.Response.Header.Get("x-request-id")
			}
		}
		var streamErr *orc.ResponsesStreamError
		if errors.As(err, &streamErr) {
			fields["status_code"] = streamErr.StatusCode
			fields["content_type"] = streamErr.ContentType
			fields["request_id"] = streamErr.RequestID
			fields["event_type"] = streamErr.EventType
			fields["response_preview"] = streamErr.Preview
		}
		logger.ErrorCF("provider.openai_responses", "OpenAI Responses API call failed", fields)
		return nil, fmt.Errorf("openai responses API call: %w", err)
	}
	return parsed, nil
}

func (p *Provider) GetDefaultModel() string {
	return ""
}

func (p *Provider) SupportsNativeSearch() bool {
	return true
}

func (p *Provider) SupportsThinking() bool {
	return true
}

func cloneHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(headers))
	for key, value := range headers {
		cloned[key] = value
	}
	return cloned
}

func buildParams(
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) responses.ResponseNewParams {
	inputItems, instructions := orc.TranslateMessages(messages)

	params := responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: inputItems,
		},
		Store: openai.Opt(false),
	}

	if instructions != "" {
		params.Instructions = openai.Opt(instructions)
	}
	if maxTokens, ok := common.AsInt(options["max_tokens"]); ok {
		params.MaxOutputTokens = openai.Opt(int64(maxTokens))
	}
	if temperature, ok := common.AsFloat(options["temperature"]); ok {
		params.Temperature = openai.Opt(temperature)
	}
	if cacheKey, ok := options["prompt_cache_key"].(string); ok && cacheKey != "" {
		params.PromptCacheKey = openai.Opt(cacheKey)
	}
	if effort, ok := reasoningEffortFromOptions(options); ok {
		params.Reasoning = shared.ReasoningParam{Effort: effort}
	}

	useNativeSearch := options["native_search"] == true
	if len(tools) > 0 || useNativeSearch {
		translatedTools := orc.TranslateTools(tools, useNativeSearch)
		if len(translatedTools) > 0 {
			params.Tools = translatedTools
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsAuto),
			}
		}
	}

	return params
}

func reasoningEffortFromOptions(options map[string]any) (shared.ReasoningEffort, bool) {
	level, ok := options["thinking_level"].(string)
	if !ok {
		return "", false
	}
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "off":
		return shared.ReasoningEffortNone, true
	case "low":
		return shared.ReasoningEffortLow, true
	case "medium":
		return shared.ReasoningEffortMedium, true
	case "high":
		return shared.ReasoningEffortHigh, true
	case "xhigh":
		return shared.ReasoningEffortXhigh, true
	default:
		return "", false
	}
}
