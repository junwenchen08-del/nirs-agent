import type { Message } from "@langchain/langgraph-sdk";
import { expect, test } from "@rstest/core";

import {
  channelSourceOfThread,
  contextOfThread,
  agentNameOfThread,
  pathOfThread,
  textOfMessage,
} from "@/core/threads/utils";

test("uses standard chat route when thread has no agent context", () => {
  expect(pathOfThread("thread-123")).toBe("/workspace/chats/thread-123");
  expect(
    pathOfThread({
      thread_id: "thread-123",
    }),
  ).toBe("/workspace/chats/thread-123");
});

test("historical agent threads use the standard route and preserve their binding", () => {
  expect(
    pathOfThread({
      thread_id: "thread-123",
      context: { agent_name: "researcher" },
    }),
  ).toBe("/workspace/chats/thread-123");
  expect(
    agentNameOfThread({
      thread_id: "thread-123",
      context: { agent_name: "researcher" },
    }),
  ).toBe("researcher");
});

test("legacy context arguments still resolve to the standard chat route", () => {
  expect(pathOfThread("thread-123", { agent_name: "ops agent" })).toBe(
    "/workspace/chats/thread-123",
  );
});

test("historical agent metadata also uses the standard chat route", () => {
  expect(
    pathOfThread({
      thread_id: "thread-456",
      metadata: { agent_name: "coder" },
    }),
  ).toBe("/workspace/chats/thread-456");
  expect(
    agentNameOfThread({
      thread_id: "thread-456",
      metadata: { agent_name: "coder" },
    }),
  ).toBe("coder");
});

test("prefers context.agent_name over metadata.agent_name", () => {
  expect(
    pathOfThread({
      thread_id: "thread-789",
      context: { agent_name: "from-context" },
      metadata: { agent_name: "from-metadata" },
    }),
  ).toBe("/workspace/chats/thread-789");
  expect(
    agentNameOfThread({
      thread_id: "thread-789",
      context: { agent_name: "from-context" },
      metadata: { agent_name: "from-metadata" },
    }),
  ).toBe("from-context");
});

test("invalid or missing historical agent metadata does not create a binding", () => {
  expect(agentNameOfThread(null)).toBeUndefined();
  expect(
    agentNameOfThread({ thread_id: "x", metadata: { agent_name: 123 } }),
  ).toBeUndefined();
  expect(
    agentNameOfThread({ thread_id: "x", metadata: { agent_name: "  " } }),
  ).toBeUndefined();
});

test("valid owned metadata survives an empty context and stale browser preferences", () => {
  const thread = {
    thread_id: "x",
    context: { agent_name: "" },
    metadata: { agent_name: "legacy-agent" },
  };
  expect(agentNameOfThread(thread)).toBe("legacy-agent");
  const preferences = {
    agent_name: "stale-agent",
    model_name: "test-model",
    thinking_enabled: true,
  };
  expect(contextOfThread(preferences, thread)).toEqual({
    agent_name: "legacy-agent",
    model_name: "test-model",
    thinking_enabled: true,
  });
  expect(contextOfThread(preferences, null)).toEqual({
    model_name: "test-model",
    thinking_enabled: true,
  });
  expect(preferences.agent_name).toBe("stale-agent");
});

test("reads IM channel source metadata", () => {
  expect(
    channelSourceOfThread({
      metadata: {
        channel_source: {
          type: "im_channel",
          provider: "feishu",
          chat_id: "oc_123",
        },
      },
    }),
  ).toEqual({
    type: "im_channel",
    provider: "feishu",
    label: "Feishu",
  });
});

test("ignores threads without valid IM channel source metadata", () => {
  expect(channelSourceOfThread({ metadata: {} })).toBeNull();
  expect(
    channelSourceOfThread({
      metadata: { channel_source: { provider: "" } },
    }),
  ).toBeNull();
  expect(
    channelSourceOfThread({
      metadata: {
        channel_source: {
          type: "other",
          provider: "feishu",
        },
      },
    }),
  ).toBeNull();
});

test("textOfMessage concatenates object and bare-string content parts", () => {
  // Gemini's finalized shape: first signed {type:text} block + bare-string
  // continuation. textOfMessage joins flat ("") for single-line consumers.
  const message = {
    id: "ai-1",
    type: "ai",
    content: [
      {
        type: "text",
        text: "First block.",
        extras: { signature: "abc123" },
        index: 0,
      },
      " Continuation as a bare string.",
    ],
  } as unknown as Message;

  expect(textOfMessage(message)).toBe(
    "First block. Continuation as a bare string.",
  );
});

test("textOfMessage returns null when array content has no text", () => {
  const message = {
    id: "ai-1",
    type: "ai",
    content: [{ type: "image_url", image_url: "https://example.com/x.png" }],
  } as unknown as Message;

  expect(textOfMessage(message)).toBeNull();
});
