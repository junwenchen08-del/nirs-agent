import type { Message } from "@langchain/langgraph-sdk";

import { extractContentFromMessage } from "./utils";

const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

export function shouldHideAutonomousReasoning(
  workflow: unknown,
  context: unknown,
): boolean {
  const settings = record(context);
  return (
    record(workflow).decision_strategy === "llm_autonomous" ||
    record(settings.nir_autonomous).enabled === true ||
    record(settings.nir_web_request).decision_strategy === "llm_autonomous"
  );
}

const reasoningKeys = new Set([
  "reasoning_content",
  "reasoning",
  "thinking",
  "reasoning_details",
]);
const withoutReasoningFields = (value: Record<string, unknown>) =>
  Object.fromEntries(
    Object.entries(value).filter(([key]) => !reasoningKeys.has(key)),
  );

/** Display copies only: never modify persisted messages, tool evidence or usage. */
export function messagesForReasoningVisibility(
  messages: Message[],
  hideRawReasoning: boolean,
): Message[] {
  if (!hideRawReasoning) return messages;
  return messages.map((message) => {
    if (message.type !== "ai") return message;
    const content =
      typeof message.content === "string"
        ? extractContentFromMessage(message)
        : message.content.flatMap<Record<string, unknown>>((part) => {
            if (typeof part === "string")
              return [
                {
                  type: "text",
                  text: extractContentFromMessage({
                    ...message,
                    content: part,
                  }),
                },
              ];
            const block = record(part);
            if (
              ["thinking", "reasoning", "redacted_thinking"].includes(
                String(block.type),
              )
            )
              return [];
            const visible = withoutReasoningFields(block);
            return [
              {
                ...visible,
                ...(block.type === "text" && typeof block.text === "string"
                  ? {
                      text: extractContentFromMessage({
                        ...message,
                        content: block.text,
                      }),
                    }
                  : {}),
              },
            ];
          });
    return {
      ...message,
      content,
      ...(message.additional_kwargs
        ? {
            additional_kwargs: withoutReasoningFields(
              message.additional_kwargs,
            ),
          }
        : {}),
    } as Message;
  });
}
