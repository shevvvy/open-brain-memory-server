const express = require("express");
const cors = require("cors");
const { createClient } = require("@supabase/supabase-js");
const OpenAI = require("openai");

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MEMORY_OWNER_UUID = process.env.MEMORY_OWNER_UUID;
const PORT = parseInt(process.env.PORT || "8080", 10);

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  throw new Error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
}

if (!OPENAI_API_KEY) {
  throw new Error("Missing OPENAI_API_KEY");
}

if (!MEMORY_OWNER_UUID) {
  throw new Error("Missing MEMORY_OWNER_UUID");
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

async function generateEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });

  return response.data[0].embedding;
}

const tools = {
  remember_note: {
    name: "remember_note",
    description:
      "Store a memory note. Use this to persist important context, decisions, people, or insights.",
    input_schema: {
      type: "object",
      properties: {
        raw_text: { type: "string" },
        source: { type: "string" },
        project: { type: "string" },
        category: { type: "string" },
        importance: { type: "number" },
        metadata: { type: "object" },
      },
      required: ["raw_text"],
      additionalProperties: false,
    },
    output_schema: {
      type: "object",
      properties: {
        id: { type: "string" },
        message: { type: "string" },
      },
      required: ["id", "message"],
    },
    readOnlyHint: false,
    run: async (args) => {
      const {
        raw_text,
        source = "manual",
        project = null,
        category = null,
        importance = 3,
        metadata = {},
      } = args;

      const embedding = await generateEmbedding(raw_text);

      const { data, error } = await supabase
        .from("notes")
        .insert({
          user_id: MEMORY_OWNER_UUID,
          raw_text,
          clean_text: raw_text,
          source,
          project,
          category,
          importance,
          metadata,
          embedding,
        })
        .select()
        .limit(1);

      if (error) {
        throw new Error(`Failed to insert note: ${error.message}`);
      }

      return {
        id: data[0].id,
        message: "Note captured successfully",
      };
    },
  },

  search_notes: {
    name: "search_notes",
    description:
      "Semantic search across notes. Use this to find related prior context by meaning.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        top_k: { type: "number" },
      },
      required: ["query"],
      additionalProperties: false,
    },
    output_schema: {
      type: "object",
      properties: {
        results: { type: "array" },
      },
      required: ["results"],
    },
    readOnlyHint: true,
    run: async (args) => {
      const { query, top_k = 5 } = args;
      const queryEmbedding = await generateEmbedding(query);

      const { data, error } = await supabase.rpc("match_notes", {
        query_embedding: queryEmbedding,
        match_count: top_k,
        user_uuid: MEMORY_OWNER_UUID,
      });

      if (error) {
        throw new Error(`Search error: ${error.message}`);
      }

      return { results: data || [] };
    },
  },

  recent_notes: {
    name: "recent_notes",
    description:
      "Return the most recent notes for the shared memory owner.",
    input_schema: {
      type: "object",
      properties: {
        limit: { type: "number" },
      },
      required: [],
      additionalProperties: false,
    },
    output_schema: {
      type: "object",
      properties: {
        results: { type: "array" },
      },
      required: ["results"],
    },
    readOnlyHint: true,
    run: async (args) => {
      const { limit = 10 } = args;

      const { data, error } = await supabase
        .from("notes")
        .select("id, raw_text, project, category, created_at")
        .eq("user_id", MEMORY_OWNER_UUID)
        .order("created_at", { ascending: false })
        .limit(limit);

      if (error) {
        throw new Error(`Failed to fetch recent notes: ${error.message}`);
      }

      return { results: data || [] };
    },
  },

  upsert_memory_fact: {
    name: "upsert_memory_fact",
    description:
      "Store a structured memory fact linked to a note.",
    input_schema: {
      type: "object",
      properties: {
        note_id: { type: "string" },
        fact_type: { type: "string" },
        subject: { type: "string" },
        value: { type: "string" },
        confidence: { type: "number" },
        status: { type: "string" },
      },
      required: ["note_id", "fact_type", "value"],
      additionalProperties: false,
    },
    output_schema: {
      type: "object",
      properties: {
        id: { type: "string" },
        message: { type: "string" },
      },
      required: ["id", "message"],
    },
    readOnlyHint: false,
    run: async (args) => {
      const {
        note_id,
        fact_type,
        subject = null,
        value,
        confidence = 0.7,
        status = "candidate",
      } = args;

      const embedding = await generateEmbedding(value);

      const { data, error } = await supabase
        .from("memory_facts")
        .insert({
          user_id: MEMORY_OWNER_UUID,
          note_id,
          fact_type,
          subject,
          value,
          confidence,
          status,
          embedding,
        })
        .select()
        .limit(1);

      if (error) {
        throw new Error(`Failed to insert memory fact: ${error.message}`);
      }

      return {
        id: data[0].id,
        message: "Memory fact stored successfully",
      };
    },
  },

  search_memory_facts: {
    name: "search_memory_facts",
    description:
      "Semantic search across memory facts.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        top_k: { type: "number" },
      },
      required: ["query"],
      additionalProperties: false,
    },
    output_schema: {
      type: "object",
      properties: {
        results: { type: "array" },
      },
      required: ["results"],
    },
    readOnlyHint: true,
    run: async (args) => {
      const { query, top_k = 5 } = args;
      const queryEmbedding = await generateEmbedding(query);

      const { data, error } = await supabase.rpc("match_memory_facts", {
        query_embedding: queryEmbedding,
        match_count: top_k,
        user_uuid: MEMORY_OWNER_UUID,
      });

      if (error) {
        throw new Error(`Search memory facts error: ${error.message}`);
      }

      return { results: data || [] };
    },
  },
};

app.get("/", (req, res) => {
  res.json({
    ok: true,
    service: "open-brain-memory-server",
    message: "Server is running",
  });
});

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

app.get("/mcp", (req, res) => {
  res.json({
    ok: true,
    message: "MCP root reachable",
    endpoints: ["/mcp/initialize", "/mcp/list_tools", "/mcp/call"],
  });
});

app.get("/mcp/initialize", (req, res) => {
  res.json({
    name: "Open Brain Memory MCP Server",
    version: "1.0.0",
    description:
      "Provides note and memory fact tools backed by Supabase with semantic search.",
    authentication: { type: "none" },
    protocol: "http",
    list_tools_endpoint: "/mcp/list_tools",
    call_endpoint: "/mcp/call",
  });
});

app.get("/mcp/list_tools", (req, res) => {
  const toolList = Object.values(tools).map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.input_schema,
    output_schema: tool.output_schema,
    readOnlyHint: !!tool.readOnlyHint,
  }));

  res.json({ tools: toolList });
});

app.post("/mcp/call", async (req, res) => {
  try {
    const toolName = req.body.tool;
    const args = req.body.arguments || {};

    if (!toolName) {
      return res.status(400).json({ error: "Missing tool name" });
    }

    const tool = tools[toolName];
    if (!tool) {
      return res.status(400).json({ error: `Unknown tool: ${toolName}` });
    }

    const required = tool.input_schema?.required || [];
    for (const key of required) {
      if (args[key] === undefined || args[key] === null) {
        return res
          .status(400)
          .json({ error: `Missing required argument: ${key}` });
      }
    }

    const result = await tool.run(args);
    return res.json({ result });
  } catch (err) {
    console.error(err);
    return res.status(500).json({
      error: err.message || "Internal server error",
    });
  }
});

app.listen(PORT, () => {
  console.log(`Open Brain Memory Server listening on port ${PORT}`);
});
