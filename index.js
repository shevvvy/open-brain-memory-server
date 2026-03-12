import express from "express";
import cors from "cors";
import { randomUUID } from "node:crypto";
import * as z from "zod/v4";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MEMORY_OWNER_UUID = process.env.MEMORY_OWNER_UUID;
const PORT = parseInt(process.env.PORT || "8080", 10);

if (!SUPABASE_URL) throw new Error("Missing SUPABASE_URL");
if (!SUPABASE_SERVICE_ROLE_KEY) throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY");
if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");
if (!MEMORY_OWNER_UUID) throw new Error("Missing MEMORY_OWNER_UUID");

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

async function generateEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text
  });

  return response.data[0].embedding;
}

function textResult(payload) {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(payload, null, 2)
      }
    ],
    structuredContent: payload
  };
}

function errorResult(message) {
  return {
    content: [
      {
        type: "text",
        text: message
      }
    ],
    isError: true
  };
}

function createMemoryServer() {
  const server = new McpServer({
    name: "open-brain-memory-server",
    version: "1.0.0"
  });

  server.registerTool(
    "remember_note",
    {
      title: "Remember note",
      description: "Store a memory note in Supabase with an embedding.",
      inputSchema: {
        raw_text: z.string().describe("The note text to store"),
        source: z.string().optional().describe("Source such as chatgpt, claude, manual"),
        project: z.string().optional().describe("Optional project tag"),
        category: z.string().optional().describe("Optional category"),
        importance: z.number().optional().describe("Importance from 1 to 5"),
        metadata: z.record(z.any()).optional().describe("Optional JSON metadata")
      },
      outputSchema: {
        id: z.string(),
        message: z.string()
      }
    },
    async ({ raw_text, source, project, category, importance, metadata }) => {
      try {
        const embedding = await generateEmbedding(raw_text);

        const { data, error } = await supabase
          .from("notes")
          .insert({
            user_id: MEMORY_OWNER_UUID,
            raw_text,
            clean_text: raw_text,
            source: source ?? "manual",
            project: project ?? null,
            category: category ?? null,
            importance: importance ?? 3,
            metadata: metadata ?? {},
            embedding
          })
          .select()
          .limit(1);

        if (error) {
          return errorResult(`Failed to insert note: ${error.message}`);
        }

        return textResult({
          id: data[0].id,
          message: "Note captured successfully"
        });
      } catch (err) {
        return errorResult(`remember_note failed: ${err.message}`);
      }
    }
  );

  server.registerTool(
    "search_notes",
    {
      title: "Search notes",
      description: "Semantic search across notes using the match_notes SQL function.",
      inputSchema: {
        query: z.string().describe("Semantic search query"),
        top_k: z.number().optional().describe("How many results to return")
      },
      outputSchema: {
        results: z.array(z.any())
      }
    },
    async ({ query, top_k }) => {
      try {
        const queryEmbedding = await generateEmbedding(query);

        const { data, error } = await supabase.rpc("match_notes", {
          query_embedding: queryEmbedding,
          match_count: top_k ?? 5,
          user_uuid: MEMORY_OWNER_UUID
        });

        if (error) {
          return errorResult(`search_notes failed: ${error.message}`);
        }

        return textResult({ results: data ?? [] });
      } catch (err) {
        return errorResult(`search_notes failed: ${err.message}`);
      }
    }
  );

  server.registerTool(
    "recent_notes",
    {
      title: "Recent notes",
      description: "List the most recent notes for the shared memory owner.",
      inputSchema: {
        limit: z.number().optional().describe("Number of notes to return")
      },
      outputSchema: {
        results: z.array(z.any())
      }
    },
    async ({ limit }) => {
      try {
        const { data, error } = await supabase
          .from("notes")
          .select("id, raw_text, project, category, created_at")
          .eq("user_id", MEMORY_OWNER_UUID)
          .order("created_at", { ascending: false })
          .limit(limit ?? 10);

        if (error) {
          return errorResult(`recent_notes failed: ${error.message}`);
        }

        return textResult({ results: data ?? [] });
      } catch (err) {
        return errorResult(`recent_notes failed: ${err.message}`);
      }
    }
  );

  server.registerTool(
    "upsert_memory_fact",
    {
      title: "Upsert memory fact",
      description: "Store a structured memory fact linked to a note.",
      inputSchema: {
        note_id: z.string().describe("The related note ID"),
        fact_type: z.string().describe("Type such as preference, decision, person"),
        subject: z.string().optional().describe("Optional subject"),
        value: z.string().describe("The fact text"),
        confidence: z.number().optional().describe("Confidence from 0 to 1"),
        status: z.string().optional().describe("Status such as candidate or approved")
      },
      outputSchema: {
        id: z.string(),
        message: z.string()
      }
    },
    async ({ note_id, fact_type, subject, value, confidence, status }) => {
      try {
        const embedding = await generateEmbedding(value);

        const { data, error } = await supabase
          .from("memory_facts")
          .insert({
            user_id: MEMORY_OWNER_UUID,
            note_id,
            fact_type,
            subject: subject ?? null,
            value,
            confidence: confidence ?? 0.7,
            status: status ?? "candidate",
            embedding
          })
          .select()
          .limit(1);

        if (error) {
          return errorResult(`upsert_memory_fact failed: ${error.message}`);
        }

        return textResult({
          id: data[0].id,
          message: "Memory fact stored successfully"
        });
      } catch (err) {
        return errorResult(`upsert_memory_fact failed: ${err.message}`);
      }
    }
  );

  server.registerTool(
    "search_memory_facts",
    {
      title: "Search memory facts",
      description: "Semantic search across memory facts using the match_memory_facts SQL function.",
      inputSchema: {
        query: z.string().describe("Semantic search query"),
        top_k: z.number().optional().describe("How many results to return")
      },
      outputSchema: {
        results: z.array(z.any())
      }
    },
    async ({ query, top_k }) => {
      try {
        const queryEmbedding = await generateEmbedding(query);

        const { data, error } = await supabase.rpc("match_memory_facts", {
          query_embedding: queryEmbedding,
          match_count: top_k ?? 5,
          user_uuid: MEMORY_OWNER_UUID
        });

        if (error) {
          return errorResult(`search_memory_facts failed: ${error.message}`);
        }

        return textResult({ results: data ?? [] });
      } catch (err) {
        return errorResult(`search_memory_facts failed: ${err.message}`);
      }
    }
  );

  return server;
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "open-brain-memory-server",
    endpoints: {
      streamable_http: "/mcp",
      sse: "/sse",
      sse_messages: "/messages",
      health: "/health"
    }
  });
});

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

const streamableSessions = new Map();
const sseSessions = new Map();

app.all("/mcp", async (req, res) => {
  try {
    const headerValue = req.headers["mcp-session-id"];
    const sessionId =
      typeof headerValue === "string"
        ? headerValue
        : Array.isArray(headerValue)
        ? headerValue[0]
        : undefined;

    let session = sessionId ? streamableSessions.get(sessionId) : undefined;

    if (!session) {
      if (req.method !== "POST" || !isInitializeRequest(req.body)) {
        return res.status(400).json({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Bad Request: No valid session ID provided"
          },
          id: null
        });
      }

      const server = createMemoryServer();

      let transport;
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (newSessionId) => {
          streamableSessions.set(newSessionId, { server, transport });
        }
      });

      transport.onclose = async () => {
        if (transport.sessionId) {
          streamableSessions.delete(transport.sessionId);
        }
        await server.close();
      };

      await server.connect(transport);
      session = { server, transport };
    }

    await session.transport.handleRequest(
      req,
      res,
      req.method === "POST" ? req.body : undefined
    );
  } catch (err) {
    console.error("Error in /mcp:", err);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32603,
          message: "Internal server error"
        },
        id: null
      });
    }
  }
});

app.get("/sse", async (_req, res) => {
  try {
    const server = createMemoryServer();
    const transport = new SSEServerTransport("/messages", res);

    sseSessions.set(transport.sessionId, { server, transport });

    transport.onclose = async () => {
      sseSessions.delete(transport.sessionId);
      await server.close();
    };

    await server.connect(transport);
  } catch (err) {
    console.error("Error in /sse:", err);
    if (!res.headersSent) {
      res.status(500).send("Failed to establish SSE connection");
    }
  }
});

app.post("/messages", async (req, res) => {
  try {
    const sessionId =
      typeof req.query.sessionId === "string"
        ? req.query.sessionId
        : Array.isArray(req.query.sessionId)
        ? req.query.sessionId[0]
        : undefined;

    if (!sessionId || !sseSessions.has(sessionId)) {
      return res.status(400).send("Invalid or missing sessionId");
    }

    const { transport } = sseSessions.get(sessionId);
    await transport.handlePostMessage(req, res, req.body);
  } catch (err) {
    console.error("Error in /messages:", err);
    if (!res.headersSent) {
      res.status(500).send("Failed to handle message");
    }
  }
});

app.listen(PORT, () => {
  console.log(`Open Brain MCP server listening on port ${PORT}`);
});
