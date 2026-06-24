package providers

import openairesponses "github.com/sipeed/picoclaw/pkg/providers/openai_responses"

type OpenAIResponsesProvider = openairesponses.Provider

func NewOpenAIResponsesProvider(
	apiKey string,
	apiBase string,
	proxy string,
	userAgent string,
	requestTimeoutSeconds int,
	customHeaders map[string]string,
) *OpenAIResponsesProvider {
	return openairesponses.NewProvider(
		apiKey,
		apiBase,
		proxy,
		userAgent,
		requestTimeoutSeconds,
		customHeaders,
	)
}
