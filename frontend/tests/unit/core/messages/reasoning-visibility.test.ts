import type { Message } from "@langchain/langgraph-sdk";
import { expect, test } from "@rstest/core";

import {
  messagesForReasoningVisibility,
  shouldHideAutonomousReasoning,
} from "@/core/messages/reasoning-visibility";
import {
  extractReasoningContentFromMessage,
  getMessageGroups,
} from "@/core/messages/utils";

const providerMessage: Message = {
  id: "ai_1",
  type: "ai",
  content: "Recorded public answer",
  additional_kwargs: {
    reasoning_content: "hidden raw reflection",
    other: "kept",
  },
  usage_metadata: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
};
test("autonomous mode hides provider reasoning before and after workflow binding", () => {
  expect(
    shouldHideAutonomousReasoning({ decision_strategy: "llm_autonomous" }, {}),
  ).toBe(true);
  expect(
    shouldHideAutonomousReasoning(undefined, {
      nir_autonomous: { enabled: true },
    }),
  ).toBe(true);
  expect(
    shouldHideAutonomousReasoning(undefined, {
      nir_web_request: { decision_strategy: "llm_autonomous" },
    }),
  ).toBe(true);
  expect(
    shouldHideAutonomousReasoning({ decision_strategy: "legacy" }, {}),
  ).toBe(false);
  expect(
    shouldHideAutonomousReasoning(undefined, {
      nir_autonomous: { enabled: "true" },
    }),
  ).toBe(false);
});
test("legacy messages and reasoning keep their existing identity and display", () => {
  const original = [providerMessage];
  expect(messagesForReasoningVisibility(original, false)).toBe(original);
  expect(extractReasoningContentFromMessage(original[0]!)).toBe(
    "hidden raw reflection",
  );
});
test("display projection removes provider reasoning without altering stored state or usage", () => {
  const [visible] = messagesForReasoningVisibility([providerMessage], true);
  expect(visible?.content).toBe("Recorded public answer");
  expect(visible?.additional_kwargs).toEqual({ other: "kept" });
  expect(visible?.type === "ai" && visible.usage_metadata).toBe(
    providerMessage.usage_metadata,
  );
  expect(extractReasoningContentFromMessage(visible!)).toBeNull();
  expect(providerMessage.additional_kwargs?.reasoning_content).toBe(
    "hidden raw reflection",
  );
});
test("inline closed and streaming think blocks remain hidden", () => {
  const messages = [
    { ...providerMessage, content: "<think>private</think>Public answer" },
    { ...providerMessage, content: "<think>private streaming" },
  ];
  expect(
    messagesForReasoningVisibility(messages, true).map(
      (message) => message.content,
    ),
  ).toEqual(["Public answer", ""]);
});
test("provider thinking blocks are removed while text, images and public output survive", () => {
  const message = {
    ...providerMessage,
    content: [
      { type: "thinking", thinking: "private" },
      { type: "reasoning", reasoning: "private" },
      { type: "text", text: "<think>private</think>Public" },
      { type: "image_url", image_url: "https://example.com/image.png" },
    ],
  } as Message;
  expect(messagesForReasoningVisibility([message], true)[0]?.content).toEqual([
    { type: "text", text: "Public" },
    { type: "image_url", image_url: "https://example.com/image.png" },
  ]);
});
test("accepted proposal reasons and real tool execution stay visible and associated", () => {
  const calls = [
    {
      id: "tool_1",
      name: "nir_submit_decision",
      args: {
        proposal: {
          rationale: "Recorded decision rationale",
          alternatives: [],
        },
      },
    },
  ];
  const ai = { ...providerMessage, content: "", tool_calls: calls } as Message;
  const tool = {
    id: "result_1",
    type: "tool",
    tool_call_id: "tool_1",
    name: "nir_submit_decision",
    content: "accepted",
  } as Message;
  const projected = messagesForReasoningVisibility([ai, tool], true);
  expect(projected[0]?.type === "ai" && projected[0].tool_calls).toBe(calls);
  expect(projected[1]).toBe(tool);
  expect(getMessageGroups(projected)[0]?.messages).toEqual(projected);
});
test("reasoning-only messages cannot create a thinking group in autonomous mode", () => {
  expect(
    getMessageGroups(
      messagesForReasoningVisibility(
        [{ ...providerMessage, content: "" }],
        true,
      ),
    ),
  ).toEqual([]);
});
test("human messages containing literal think tags remain unchanged", () => {
  const human = {
    id: "human_1",
    type: "human",
    content: "Explain <think> tags",
  } as Message;
  expect(messagesForReasoningVisibility([human], true)[0]).toBe(human);
});
