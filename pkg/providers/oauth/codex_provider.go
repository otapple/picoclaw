package oauthprovider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"

	"github.com/sipeed/picoclaw/pkg/auth"
	"github.com/sipeed/picoclaw/pkg/logger"
	orc "github.com/sipeed/picoclaw/pkg/providers/openai_responses_common"
)

const (
	codexDefaultModel        = "gpt-5.3-codex"
	codexDefaultInstructions = "You are Codex, a coding assistant."
)

type CodexProvider struct {
	httpClient      *http.Client
	apiBase         string
	apiKey          string
	accountID       string
	baseHeaders     map[string]string
	tokenSource     func() (string, string, error)
	enableWebSearch bool
}

const defaultCodexInstructions = "You are Codex, a coding assistant."

func NewCodexProvider(token, accountID string) *CodexProvider {
	return &CodexProvider{
		httpClient: http.DefaultClient,
		apiBase:    "https://chatgpt.com/backend-api/codex",
		apiKey:     token,
		accountID:  accountID,
		baseHeaders: map[string]string{
			"originator":  "codex_cli_rs",
			"OpenAI-Beta": "responses=experimental",
		},
		enableWebSearch: true,
	}
}

func NewCodexProviderWithTokenSource(
	token, accountID string, tokenSource func() (string, string, error),
) *CodexProvider {
	p := NewCodexProvider(token, accountID)
	p.tokenSource = tokenSource
	return p
}

func (p *CodexProvider) Chat(
	ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]any,
) (*LLMResponse, error) {
	if p == nil || p.httpClient == nil {
		return nil, fmt.Errorf("codex provider is not configured")
	}
	apiKey := p.apiKey
	accountID := p.accountID
	resolvedModel, fallbackReason := resolveCodexModel(model)
	if fallbackReason != "" {
		logger.WarnCF(
			"provider.codex",
			"Requested model is not compatible with Codex backend, using fallback",
			map[string]any{
				"requested_model": model,
				"resolved_model":  resolvedModel,
				"reason":          fallbackReason,
			},
		)
	}
	if p.tokenSource != nil {
		tok, accID, err := p.tokenSource()
		if err != nil {
			return nil, fmt.Errorf("refreshing token: %w", err)
		}
		apiKey = tok
		if accID != "" {
			accountID = accID
		}
	}
	headers := cloneStringMap(p.baseHeaders)
	if accountID != "" {
		headers["Chatgpt-Account-Id"] = accountID
	} else {
		logger.WarnCF(
			"provider.codex",
			"No account id found for Codex request; backend may reject with 400",
			map[string]any{
				"requested_model": model,
				"resolved_model":  resolvedModel,
			},
		)
	}

	// Respect tools.web.prefer_native: only inject native search when the agent
	// loop passes options["native_search"]=true, so prefer_native=false means no injection.
	useNativeSearch := p.enableWebSearch && (options["native_search"] == true)
	params := buildCodexParams(messages, tools, resolvedModel, options, useNativeSearch)

	parsed, err := orc.DoStreamingResponseRequest(ctx, orc.StreamingRequest{
		HTTPClient: p.httpClient,
		APIBase:    p.apiBase,
		APIKey:     apiKey,
		Headers:    headers,
		Params:     params,
	})
	if err != nil {
		fields := map[string]any{
			"requested_model":    model,
			"resolved_model":     resolvedModel,
			"messages_count":     len(messages),
			"tools_count":        len(tools),
			"account_id_present": accountID != "",
			"error":              err.Error(),
		}
		var apiErr *openai.Error
		if errors.As(err, &apiErr) {
			fields["status_code"] = apiErr.StatusCode
			fields["api_type"] = apiErr.Type
			fields["api_code"] = apiErr.Code
			fields["api_param"] = apiErr.Param
			fields["api_message"] = apiErr.Message
			if apiErr.StatusCode == 400 {
				fields["hint"] = "verify account id header and model compatibility for codex backend"
			}
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
		logger.ErrorCF("provider.codex", "Codex API call failed", fields)
		return nil, fmt.Errorf("codex API call: %w", err)
	}
	return parsed, nil
}

func (p *CodexProvider) GetDefaultModel() string {
	return codexDefaultModel
}

func (p *CodexProvider) SupportsNativeSearch() bool {
	return p.enableWebSearch
}

func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return map[string]string{}
	}
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func resolveCodexModel(model string) (string, string) {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return codexDefaultModel, "empty model"
	}

	if after, ok := strings.CutPrefix(m, "openai/"); ok {
		m = after
	} else if strings.Contains(m, "/") {
		return codexDefaultModel, "non-openai model namespace"
	}

	unsupportedPrefixes := []string{
		"glm",
		"claude",
		"anthropic",
		"gemini",
		"google",
		"moonshot",
		"kimi",
		"qwen",
		"deepseek",
		"llama",
		"meta-llama",
		"mistral",
		"grok",
		"xai",
		"zhipu",
	}
	for _, prefix := range unsupportedPrefixes {
		if strings.HasPrefix(m, prefix) {
			return codexDefaultModel, "unsupported model prefix"
		}
	}

	if strings.HasPrefix(m, "gpt-") || strings.HasPrefix(m, "o3") || strings.HasPrefix(m, "o4") {
		return m, ""
	}

	return codexDefaultModel, "unsupported model family"
}

func buildCodexParams(
	messages []Message, tools []ToolDefinition, model string, options map[string]any, enableWebSearch bool,
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
	} else {
		// ChatGPT Codex backend requires instructions to be present.
		params.Instructions = openai.Opt(defaultCodexInstructions)
	}

	// Prompt caching: pass a stable cache key so OpenAI can bucket requests
	// and reuse prefix KV cache across calls with the same key.
	// See: https://platform.openai.com/docs/guides/prompt-caching
	if cacheKey, ok := options["prompt_cache_key"].(string); ok && cacheKey != "" {
		params.PromptCacheKey = openai.Opt(cacheKey)
	}

	if len(tools) > 0 || enableWebSearch {
		params.Tools = orc.TranslateTools(tools, enableWebSearch)
	}

	return params
}

func CreateCodexTokenSource() func() (string, string, error) {
	return func() (string, string, error) {
		cred, err := auth.GetCredential("openai")
		if err != nil {
			return "", "", fmt.Errorf("loading auth credentials: %w", err)
		}
		if cred == nil {
			return "", "", fmt.Errorf("no credentials for openai. Run: picoclaw auth login --provider openai")
		}

		if cred.AuthMethod == "oauth" && cred.NeedsRefresh() && cred.RefreshToken != "" {
			oauthCfg := auth.OpenAIOAuthConfig()
			refreshed, err := auth.RefreshAccessToken(cred, oauthCfg)
			if err != nil {
				return "", "", fmt.Errorf("refreshing token: %w", err)
			}
			if refreshed.AccountID == "" {
				refreshed.AccountID = cred.AccountID
			}
			if err := auth.SetCredential("openai", refreshed); err != nil {
				return "", "", fmt.Errorf("saving refreshed token: %w", err)
			}
			return refreshed.AccessToken, refreshed.AccountID, nil
		}

		return cred.AccessToken, cred.AccountID, nil
	}
}
