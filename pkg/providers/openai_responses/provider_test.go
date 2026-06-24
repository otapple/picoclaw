package openai_responses

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
)

func TestBuildParams_BasicOptions(t *testing.T) {
	params := buildParams(
		[]Message{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "Hello"},
		},
		nil,
		"gpt-5.4",
		map[string]any{
			"max_tokens":       2048,
			"temperature":      0.4,
			"prompt_cache_key": "cache-key",
		},
	)

	if params.Model != "gpt-5.4" {
		t.Fatalf("Model = %q, want gpt-5.4", params.Model)
	}
	if !params.Instructions.Valid() || params.Instructions.Or("") != "You are helpful" {
		t.Fatalf("Instructions = %q, want system instructions", params.Instructions.Or(""))
	}
	if !params.Store.Valid() || params.Store.Or(true) {
		t.Fatalf("Store = %v, want false", params.Store.Or(true))
	}
	if !params.MaxOutputTokens.Valid() || params.MaxOutputTokens.Or(0) != 2048 {
		t.Fatalf("MaxOutputTokens = %d, want 2048", params.MaxOutputTokens.Or(0))
	}
	if !params.Temperature.Valid() || params.Temperature.Or(0) != 0.4 {
		t.Fatalf("Temperature = %f, want 0.4", params.Temperature.Or(0))
	}
	if !params.PromptCacheKey.Valid() || params.PromptCacheKey.Or("") != "cache-key" {
		t.Fatalf("PromptCacheKey = %q, want cache-key", params.PromptCacheKey.Or(""))
	}
}

func TestBuildParams_NoCodexDefaultInstructions(t *testing.T) {
	params := buildParams(
		[]Message{{Role: "user", Content: "Hello"}},
		nil,
		"gpt-5.4",
		map[string]any{},
	)

	if params.Instructions.Valid() {
		t.Fatalf("Instructions should be omitted when there is no system message, got %q", params.Instructions.Or(""))
	}
}

func TestBuildParams_NativeWebSearchInjection(t *testing.T) {
	tools := []ToolDefinition{
		{
			Type: "function",
			Function: ToolFunctionDefinition{
				Name:       "web_search",
				Parameters: map[string]any{"type": "object"},
			},
		},
		{
			Type: "function",
			Function: ToolFunctionDefinition{
				Name:       "read_file",
				Parameters: map[string]any{"type": "object"},
			},
		},
	}

	params := buildParams(
		[]Message{{Role: "user", Content: "Hi"}},
		tools,
		"gpt-5.4",
		map[string]any{"native_search": true},
	)

	if len(params.Tools) != 2 {
		t.Fatalf("len(Tools) = %d, want 2", len(params.Tools))
	}
	if params.Tools[0].OfFunction == nil || params.Tools[0].OfFunction.Name != "read_file" {
		t.Fatalf("first tool should be read_file function, got %#v", params.Tools[0])
	}
	if params.Tools[1].OfWebSearch == nil {
		t.Fatalf("second tool should be built-in web_search, got %#v", params.Tools[1])
	}
	if params.Tools[1].OfWebSearch.Type != responses.WebSearchToolTypeWebSearch {
		t.Fatalf("web search type = %q, want web_search", params.Tools[1].OfWebSearch.Type)
	}
	if !params.ToolChoice.OfToolChoiceMode.Valid() ||
		params.ToolChoice.OfToolChoiceMode.Or("") != responses.ToolChoiceOptionsAuto {
		t.Fatalf("ToolChoice mode = %q, want auto", params.ToolChoice.OfToolChoiceMode.Or(""))
	}
}

func TestBuildParams_UserWebSearchFunctionWhenNativeDisabled(t *testing.T) {
	tools := []ToolDefinition{
		{
			Type: "function",
			Function: ToolFunctionDefinition{
				Name:       "web_search",
				Parameters: map[string]any{"type": "object"},
			},
		},
	}

	params := buildParams(
		[]Message{{Role: "user", Content: "Hi"}},
		tools,
		"gpt-5.4",
		map[string]any{},
	)

	if len(params.Tools) != 1 {
		t.Fatalf("len(Tools) = %d, want 1", len(params.Tools))
	}
	if params.Tools[0].OfFunction == nil || params.Tools[0].OfFunction.Name != "web_search" {
		t.Fatalf("tool should be user-defined web_search function, got %#v", params.Tools[0])
	}
}

func TestBuildParams_OmitsToolChoiceWhenNoToolsSurviveTranslation(t *testing.T) {
	params := buildParams(
		[]Message{{Role: "user", Content: "Hi"}},
		[]ToolDefinition{{Type: "not_function"}},
		"gpt-5.4",
		map[string]any{},
	)

	if len(params.Tools) != 0 {
		t.Fatalf("len(Tools) = %d, want 0", len(params.Tools))
	}
	if params.ToolChoice.OfToolChoiceMode.Valid() {
		t.Fatalf("ToolChoice should be omitted when no tools are sent, got %q", params.ToolChoice.OfToolChoiceMode.Or(""))
	}
}

func TestBuildParams_ThinkingLevelReasoningEffort(t *testing.T) {
	tests := []struct {
		level    string
		want     string
		wantSent bool
	}{
		{level: "off", want: "none", wantSent: true},
		{level: "low", want: "low", wantSent: true},
		{level: "medium", want: "medium", wantSent: true},
		{level: "high", want: "high", wantSent: true},
		{level: "xhigh", want: "xhigh", wantSent: true},
		{level: "adaptive", wantSent: false},
		{level: "invalid", wantSent: false},
	}

	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			params := buildParams(
				[]Message{{Role: "user", Content: "Hi"}},
				nil,
				"gpt-5.4",
				map[string]any{"thinking_level": tt.level},
			)
			data, err := json.Marshal(params)
			if err != nil {
				t.Fatalf("Marshal params: %v", err)
			}
			var body map[string]any
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatalf("Unmarshal params: %v", err)
			}
			reasoning, sent := body["reasoning"].(map[string]any)
			if sent != tt.wantSent {
				t.Fatalf("reasoning sent = %v, want %v; body=%s", sent, tt.wantSent, string(data))
			}
			if tt.wantSent && reasoning["effort"] != tt.want {
				t.Fatalf("reasoning.effort = %v, want %q", reasoning["effort"], tt.want)
			}
		})
	}
}

func TestProvider_ChatRoundTrip(t *testing.T) {
	var capturedPath string
	var capturedAuth string
	var capturedUserAgent string
	var capturedCustom string
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedAuth = r.Header.Get("Authorization")
		capturedUserAgent = r.Header.Get("User-Agent")
		capturedCustom = r.Header.Get("X-Source")
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		if requestBody["stream"] != true {
			http.Error(w, "stream must be true", http.StatusBadRequest)
			return
		}
		writeCompletedSSE(w, map[string]any{
			"id":     "resp_test",
			"object": "response",
			"status": "completed",
			"output": []map[string]any{
				{
					"id":     "msg_1",
					"type":   "message",
					"role":   "assistant",
					"status": "completed",
					"content": []map[string]any{
						{"type": "output_text", "text": "Hello from Responses"},
					},
				},
			},
			"usage": map[string]any{
				"input_tokens":          5,
				"output_tokens":         3,
				"total_tokens":          8,
				"input_tokens_details":  map[string]any{"cached_tokens": 0},
				"output_tokens_details": map[string]any{"reasoning_tokens": 0},
			},
		})
	}))
	defer server.Close()

	provider := NewProvider(
		"test-key",
		server.URL,
		"",
		"UnitTest/1.0",
		0,
		map[string]string{"X-Source": "picoclaw-test"},
	)
	resp, err := provider.Chat(
		t.Context(),
		[]Message{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "Hello"},
		},
		nil,
		"gpt-5.4",
		map[string]any{
			"max_tokens":       512,
			"temperature":      0.2,
			"prompt_cache_key": "test-cache",
			"thinking_level":   "high",
		},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if capturedPath != "/responses" {
		t.Fatalf("path = %q, want /responses", capturedPath)
	}
	if capturedAuth != "Bearer test-key" {
		t.Fatalf("Authorization = %q, want bearer key", capturedAuth)
	}
	if capturedUserAgent != "UnitTest/1.0" {
		t.Fatalf("User-Agent = %q, want UnitTest/1.0", capturedUserAgent)
	}
	if capturedCustom != "picoclaw-test" {
		t.Fatalf("X-Source = %q, want picoclaw-test", capturedCustom)
	}
	if requestBody["model"] != "gpt-5.4" {
		t.Fatalf("model = %v, want gpt-5.4", requestBody["model"])
	}
	if requestBody["instructions"] != "You are helpful" {
		t.Fatalf("instructions = %v, want system prompt", requestBody["instructions"])
	}
	if requestBody["store"] != false {
		t.Fatalf("store = %v, want false", requestBody["store"])
	}
	if requestBody["max_output_tokens"] != float64(512) {
		t.Fatalf("max_output_tokens = %v, want 512", requestBody["max_output_tokens"])
	}
	if requestBody["temperature"] != 0.2 {
		t.Fatalf("temperature = %v, want 0.2", requestBody["temperature"])
	}
	if requestBody["prompt_cache_key"] != "test-cache" {
		t.Fatalf("prompt_cache_key = %v, want test-cache", requestBody["prompt_cache_key"])
	}
	reasoning, ok := requestBody["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %#v, want effort high", requestBody["reasoning"])
	}
	if resp.Content != "Hello from Responses" {
		t.Fatalf("Content = %q, want response text", resp.Content)
	}
	if resp.Usage.TotalTokens != 8 {
		t.Fatalf("TotalTokens = %d, want 8", resp.Usage.TotalTokens)
	}
}

func TestProvider_ChatRoundTrip_OutputTextDeltaFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeOutputTextDeltaSSE(w, "OK", map[string]any{
			"id":     "resp_test",
			"object": "response",
			"status": "completed",
			"output": nil,
		})
	}))
	defer server.Close()

	provider := NewProvider("test-key", server.URL, "", "", 0, nil)
	resp, err := provider.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "Hello"}},
		nil,
		"gpt-5.4",
		map[string]any{},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.Content != "OK" {
		t.Fatalf("Content = %q, want OK", resp.Content)
	}
}

func TestProvider_ChatRoundTrip_SkipsHeartbeatFrames(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeHeartbeatThenCompletedSSE(w, map[string]any{
			"id":     "resp_test",
			"object": "response",
			"status": "completed",
			"output": []map[string]any{
				{
					"id":     "msg_1",
					"type":   "message",
					"role":   "assistant",
					"status": "completed",
					"content": []map[string]any{
						{"type": "output_text", "text": "heartbeat ok"},
					},
				},
			},
		})
	}))
	defer server.Close()

	provider := NewProvider("test-key", server.URL, "", "", 0, nil)
	resp, err := provider.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "Hello"}},
		nil,
		"gpt-5.4",
		map[string]any{},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.Content != "heartbeat ok" {
		t.Fatalf("Content = %q, want heartbeat ok", resp.Content)
	}
}

func TestProvider_ChatRoundTrip_MalformedSSEHasContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("x-request-id", "req_bad")
		fmt.Fprintf(w, "event: response.completed\n")
		fmt.Fprintf(w, "data: {\"type\":\"response.completed\"\n\n")
	}))
	defer server.Close()

	provider := NewProvider("test-key", server.URL, "", "", 0, nil)
	_, err := provider.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "Hello"}},
		nil,
		"gpt-5.4",
		map[string]any{},
	)
	if err == nil {
		t.Fatal("expected malformed SSE error")
	}
	errText := err.Error()
	if !strings.Contains(errText, "response.completed") || !strings.Contains(errText, "req_bad") {
		t.Fatalf("error = %q, want event type and request id context", errText)
	}
}

func TestProvider_ChatRoundTrip_OutputItemDoneFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeOutputItemDoneSSE(
			w,
			map[string]any{
				"id":        "fc_1",
				"type":      "function_call",
				"call_id":   "call_abc",
				"name":      "write_file",
				"arguments": `{"path":"x.txt","content":"ok"}`,
				"status":    "completed",
			},
			map[string]any{
				"id":     "resp_test",
				"object": "response",
				"status": "completed",
				"output": []map[string]any{},
			},
		)
	}))
	defer server.Close()

	provider := NewProvider("test-key", server.URL, "", "", 0, nil)
	resp, err := provider.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "Create x.txt"}},
		[]ToolDefinition{
			{
				Type: "function",
				Function: ToolFunctionDefinition{
					Name:       "write_file",
					Parameters: map[string]any{"type": "object"},
				},
			},
		},
		"gpt-5.4",
		map[string]any{},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_abc" || tc.Name != "write_file" {
		t.Fatalf("tool call = %#v, want write_file call_abc", tc)
	}
	if tc.Arguments["path"] != "x.txt" || tc.Arguments["content"] != "ok" {
		t.Fatalf("tool call arguments = %#v", tc.Arguments)
	}
}

func TestProvider_MissingAPIKey(t *testing.T) {
	provider := NewProvider("", "https://api.openai.com/v1", "", "", 0, nil)
	_, err := provider.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "Hello"}},
		nil,
		"gpt-5.4",
		map[string]any{},
	)
	if err == nil {
		t.Fatal("expected missing API key error")
	}
}

func writeCompletedSSE(w http.ResponseWriter, response map[string]any) {
	event := map[string]any{
		"type":            "response.completed",
		"sequence_number": 1,
		"response":        response,
	}
	b, _ := json.Marshal(event)
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(w, "event: response.completed\n")
	fmt.Fprintf(w, "data: %s\n\n", string(b))
	fmt.Fprintf(w, "data: [DONE]\n\n")
}

func writeHeartbeatThenCompletedSSE(w http.ResponseWriter, response map[string]any) {
	event := map[string]any{
		"type":            "response.completed",
		"sequence_number": 1,
		"response":        response,
	}
	b, _ := json.Marshal(event)
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(w, ": keep-alive\n\n")
	fmt.Fprintf(w, "event: ping\n\n")
	fmt.Fprintf(w, "event: response.completed\n")
	fmt.Fprintf(w, "data: %s\n\n", string(b))
	fmt.Fprintf(w, "data: [DONE]\n\n")
}

func writeOutputTextDeltaSSE(w http.ResponseWriter, delta string, response map[string]any) {
	deltaEvent := map[string]any{
		"type":            "response.output_text.delta",
		"sequence_number": 1,
		"delta":           delta,
	}
	completedEvent := map[string]any{
		"type":            "response.completed",
		"sequence_number": 2,
		"response":        response,
	}
	deltaBytes, _ := json.Marshal(deltaEvent)
	completedBytes, _ := json.Marshal(completedEvent)
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(w, "event: response.output_text.delta\n")
	fmt.Fprintf(w, "data: %s\n\n", string(deltaBytes))
	fmt.Fprintf(w, "event: response.completed\n")
	fmt.Fprintf(w, "data: %s\n\n", string(completedBytes))
	fmt.Fprintf(w, "data: [DONE]\n\n")
}

func writeOutputItemDoneSSE(w http.ResponseWriter, item map[string]any, response map[string]any) {
	itemEvent := map[string]any{
		"type":            "response.output_item.done",
		"sequence_number": 1,
		"output_index":    0,
		"item":            item,
	}
	completedEvent := map[string]any{
		"type":            "response.completed",
		"sequence_number": 2,
		"response":        response,
	}
	itemBytes, _ := json.Marshal(itemEvent)
	completedBytes, _ := json.Marshal(completedEvent)
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(w, "event: response.output_item.done\n")
	fmt.Fprintf(w, "data: %s\n\n", string(itemBytes))
	fmt.Fprintf(w, "event: response.completed\n")
	fmt.Fprintf(w, "data: %s\n\n", string(completedBytes))
	fmt.Fprintf(w, "data: [DONE]\n\n")
}
