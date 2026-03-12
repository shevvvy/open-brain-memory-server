// Express-based remote MCP server for Open Brain memory
//
// This server implements a simple Model Context Protocol (MCP) interface
// with three endpoints:
// - GET /mcp/initialize: returns basic server metadata and information about
//   authentication and tooling endpoints.
// - GET /mcp/list_tools: returns a list of tool definitions available to
//   clients, including names, descriptions, parameter schemas, and read-only hints.
// - POST /mcp/call: executes a specified tool with provided arguments and
//   returns the result.
//
// Each tool connects to a Supabase database to either store or retrieve
// information about notes and structured memory facts. Embeddings are
// generated on demand using OpenAI's embeddings API.

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { createClient } = require('@supabase/supabase-js');
const { Configuration, OpenAIApi } = require('openai');

// Load configuration from environment variables
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MEMORY_OWNER_UUID = process.env.MEMORY_OWNER_UUID;
const PORT = parseInt(process.env.PORT, 10) || 8080;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.warn('Warning: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be provided');
}

if (!OPENAI_API_KEY) {
  console.warn('Warning: OPENAI_API_KEY is not set. Embedding generation will fail.');
}

if (!MEMORY_OWNER_UUID) {
  console.warn('Warning: MEMORY_OWNER_UUID is not set. Using a fixed default.');
}

// Initialize Supabase client using service role key
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Initialize OpenAI API client for embeddings
const openai = new OpenAIApi(new Configuration({ apiKey: OPENAI_API_KEY }));

// Helper to generate embedding for a given text using OpenAI
async function generateEmbedding(text) {
  // Fallback: return an empty array if API key is not provided
  if (!OPENAI_API_KEY) {
    return [];
  }
  const response = await openai.createEmbedding({
    model: 'text-embedding-3-small',
    input: text,
  });
  const embedding = response.data.data[0].embedding;
  return embedding;
}

// Define tool implementations
const tools = {
  remember_note: {
    name: 'remember_note',
    description:
      'Store a memory note. Use this to capture thoughts or information that should be persisted. It will generate semantic embeddings automatically.',
    input_schema: {
      type: 'object',
      properties: {
        raw_text: { type: 'string', description: 'The raw text of the note to remember.' },
        source: { type: 'string', description: 'The source of the note (e.g. chatgpt, claude).', nullable: true },
        project: { type: 'string', description: 'Optional project or tag associated with the note.', nullable: true },
        category: { type: 'string', description: 'Optional category of the note.', nullable: true },
        importance: { type: 'number', description: 'Importance level from 1 to 5.', nullable: true },
        metadata: { type: 'object', description: 'Arbitrary JSON metadata associated with the note.', nullable: true },
      },
      required: ['raw_text'],
      additionalProperties: false,
    },
    output_schema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'The UUID of the stored note.' },
        message: { type: 'string', description: 'A confirmation message.' },
      },
      required: ['id', 'message'],
    },
    readOnlyHint: false,
    run: async (args) => {
      const {
        raw_text,
        source = 'manual',
        project = null,
        category = null,
        importance = 3,
        metadata = {},
      } = args;
      // Generate embedding from the raw text
      const embedding = await generateEmbedding(raw_text);
      // Insert into Supabase notes table
      const { data, error } = await supabase.from('notes').insert({
        user_id: MEMORY_OWNER_UUID,
        raw_text: raw_text,
        clean_text: raw_text,
        source,
        project,
        category,
        importance,
        metadata,
        embedding,
      }).select();
      if (error) {
        throw new Error('Failed to insert note: ' + error.message);
      }
      const inserted = data && data[0];
      return { id: inserted.id, message: 'Note captured successfully' };
    },
  },
  search_notes: {
    name: 'search_notes',
    description:
      'Semantic search across notes. Provide a free-text query and retrieve the most similar notes based on meaning.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query text.' },
        top_k: { type: 'number', description: 'Number of top similar notes to return.', nullable: true },
      },
      required: ['query'],
      additionalProperties: false,
    },
    output_schema: {
      type: 'object',
      properties: {
        results: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              raw_text: { type: 'string' },
              project: { type: ['string', 'null'] },
              category: { type: ['string', 'null'] },
              similarity: { type: 'number' },
              created_at: { type: 'string' },
            },
            required: ['id', 'raw_text', 'similarity', 'created_at'],
          },
        },
      },
      required: ['results'],
    },
    readOnlyHint: true,
    run: async (args) => {
      const { query, top_k = 5 } = args;
      const embedding = await generateEmbedding(query);
      const { data, error } = await supabase.rpc('match_notes', {
        query_embedding: embedding,
        match_count: top_k,
        user_uuid: MEMORY_OWNER_UUID,
      });
      if (error) {
        throw new Error('Search error: ' + error.message);
      }
      // The RPC returns { id, raw_text, project, category, metadata, created_at, similarity }
      return { results: data };
    },
  },
  recent_notes: {
    name: 'recent_notes',
    description:
      'List the most recent notes stored for the current memory owner. Useful for browsing recent thoughts.',
    input_schema: {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Number of most recent notes to return.', nullable: true },
      },
      required: [],
      additionalProperties: false,
    },
    output_schema: {
      type: 'object',
      properties: {
        results: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              raw_text: { type: 'string' },
              project: { type: ['string', 'null'] },
              category: { type: ['string', 'null'] },
              created_at: { type: 'string' },
            },
            required: ['id', 'raw_text', 'created_at'],
          },
        },
      },
      required: ['results'],
    },
    readOnlyHint: true,
    run: async (args) => {
      const { limit = 10 } = args;
      const { data, error } = await supabase
        .from('notes')
        .select('id, raw_text, project, category, created_at')
        .eq('user_id', MEMORY_OWNER_UUID)
        .order('created_at', { ascending: false })
        .limit(limit);
      if (error) {
        throw new Error('Failed to fetch recent notes: ' + error.message);
      }
      return { results: data };
    },
  },
  upsert_memory_fact: {
    name: 'upsert_memory_fact',
    description:
      'Create or update a structured fact associated with a note. Use this to promote important decisions or insights into a stable memory.',
    input_schema: {
      type: 'object',
      properties: {
        note_id: { type: 'string', description: 'The ID of the note this fact is related to.' },
        fact_type: { type: 'string', description: 'The type of fact (e.g. person, decision, task, preference, insight).' },
        subject: { type: ['string', 'null'], description: 'Optional subject that this fact is about.' },
        value: { type: 'string', description: 'The content or value of the fact.' },
        confidence: { type: 'number', description: 'Confidence score between 0 and 1.', nullable: true },
        status: { type: 'string', description: "Fact status (e.g. 'candidate', 'approved', 'archived')", nullable: true },
      },
      required: ['note_id', 'fact_type', 'value'],
      additionalProperties: false,
    },
    output_schema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'The ID of the upserted fact.' },
        message: { type: 'string', description: 'A confirmation message.' },
      },
      required: ['id', 'message'],
    },
    readOnlyHint: false,
    run: async (args) => {
      const {
        note_id,
        fact_type,
        subject = null,
        value,
        confidence = 0.7,
        status = 'candidate',
      } = args;
      const embedding = await generateEmbedding(value);
      const { data, error } = await supabase
        .from('memory_facts')
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
        .select();
      if (error) {
        throw new Error('Failed to upsert memory fact: ' + error.message);
      }
      const inserted = data && data[0];
      return { id: inserted.id, message: 'Memory fact stored successfully' };
    },
  },
  search_memory_facts: {
    name: 'search_memory_facts',
    description:
      'Semantic search across memory facts. Provide a query and retrieve facts that are conceptually similar.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Query text to search for similar facts.' },
        top_k: { type: 'number', description: 'Number of similar facts to return.', nullable: true },
      },
      required: ['query'],
      additionalProperties: false,
    },
    output_schema: {
      type: 'object',
      properties: {
        results: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              note_id: { type: 'string' },
              fact_type: { type: 'string' },
              subject: { type: ['string', 'null'] },
              value: { type: 'string' },
              similarity: { type: 'number' },
            },
            required: ['id', 'note_id', 'fact_type', 'value', 'similarity'],
          },
        },
      },
      required: ['results'],
    },
    readOnlyHint: true,
    run: async (args) => {
      const { query, top_k = 5 } = args;
      const embedding = await generateEmbedding(query);
      const { data, error } = await supabase.rpc('match_memory_facts', {
        query_embedding: embedding,
        match_count: top_k,
        user_uuid: MEMORY_OWNER_UUID,
      });
      if (error) {
        throw new Error('Search memory facts error: ' + error.message);
      }
      return { results: data };
    },
  },
};

// Initialize express app
const app = express();
app.use(cors());
app.use(bodyParser.json());

// MCP initialize route
app.get('/mcp/initialize', (req, res) => {
  // Provide high-level information about the server. ChatGPT uses this to
  // determine protocol support and endpoints. We specify that the server uses
  // HTTP endpoints for tool discovery and execution, and no authentication.
  res.json({
    name: 'Open Brain Memory MCP Server',
    version: '1.0.0',
    description:
      'Provides note and memory fact tools backed by a Supabase database with semantic search and storage.',
    authentication: { type: 'none' },
    protocol: 'http',
    list_tools_endpoint: '/mcp/list_tools',
    call_endpoint: '/mcp/call',
  });
});

// MCP list_tools route
app.get('/mcp/list_tools', (req, res) => {
  const toolList = Object.keys(tools).map((key) => {
    const t = tools[key];
    return {
      name: t.name,
      description: t.description,
      input_schema: t.input_schema,
      output_schema: t.output_schema,
      // Indicate readOnlyHint for ChatGPT: true means read-only (no write operations)
      readOnlyHint: !!t.readOnlyHint,
    };
  });
  res.json({ tools: toolList });
});

// MCP call route
app.post('/mcp/call', async (req, res) => {
  try {
    const { tool, arguments: args } = req.body;
    if (!tool) {
      return res.status(400).json({ error: 'Missing tool name in request' });
    }
    const t = tools[tool];
    if (!t) {
      return res.status(400).json({ error: `Unknown tool: ${tool}` });
    }
    // Validate arguments: ensure required keys exist
    const inputSchema = t.input_schema;
    if (inputSchema && inputSchema.required) {
      for (const prop of inputSchema.required) {
        if (args[prop] === undefined || args[prop] === null) {
          return res.status(400).json({ error: `Missing required argument: ${prop}` });
        }
      }
    }
    // Execute tool
    const result = await t.run(args || {});
    res.json({ result });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Internal error' });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Memory MCP server listening on port ${PORT}`);
});
