import express from 'express';
import fetch from 'node-fetch';
import { createClient } from '@supabase/supabase-js';

// Environment variables
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const openaiKey = process.env.OPENAI_API_KEY;
const memoryOwnerUUID = process.env.MEMORY_OWNER_UUID;

if (!supabaseUrl || !supabaseKey || !openaiKey || !memoryOwnerUUID) {
  console.error('Missing environment variables');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

async function embedText(text) {
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + openaiKey,
    },
    body: JSON.stringify({
      input: text,
      model: 'text-embedding-3-small',
    }),
  });
  if (!response.ok) {
    throw new Error('OpenAI error: ' + await response.text());
  }
  const data = await response.json();
  return data.data[0].embedding;
}

app.post('/remember_note', async (req, res) => {
  const { raw_text, source, project, category, importance, metadata } = req.body;
  if (!raw_text) {
    return res.status(400).json({ error: 'raw_text is required' });
  }
  try {
    const embedding = await embedText(raw_text);
    const { error } = await supabase.from('notes').insert({
      user_id: memoryOwnerUUID,
      raw_text,
      clean_text: raw_text,
      source: source || 'manual',
      project: project || null,
      category: category || null,
      importance: importance || 3,
      metadata: metadata || {},
      embedding,
    });
    if (error) {
      return res.status(500).json({ error: error.message });
    }
    return res.status(200).json({ message: 'Note saved' });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
});

app.post('/search_notes', async (req, res) => {
  const { query, top_k } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'query is required' });
  }
  try {
    const queryEmbedding = await embedText(query);
    const { data, error } = await supabase.rpc('match_notes', {
      query_embedding: queryEmbedding,
      match_count: top_k || 5,
      user_uuid: memoryOwnerUUID,
    });
    if (error) {
      return res.status(500).json({ error: error.message });
    }
    return res.status(200).json({ results: data });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
});

app.get('/recent_notes', async (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  try {
    const { data, error } = await supabase
      .from('notes')
      .select('*')
      .eq('user_id', memoryOwnerUUID)
      .order('created_at', { ascending: false })
      .limit(limit);
    if (error) {
      return res.status(500).json({ error: error.message });
    }
    return res.status(200).json({ results: data });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
});

app.post('/upsert_memory_fact', async (req, res) => {
  const { note_id, fact_type, subject, value, confidence, status, embedding } = req.body;
  if (!note_id || !fact_type || !value) {
    return res.status(400).json({ error: 'note_id, fact_type and value are required' });
  }
  try {
    let factEmbedding = embedding;
    if (!embedding && value) {
      factEmbedding = await embedText(value);
    }
    const { error } = await supabase.from('memory_facts').insert({
      user_id: memoryOwnerUUID,
      note_id,
      fact_type,
      subject,
      value,
      confidence: confidence || null,
      status: status || 'candidate',
      embedding: factEmbedding,
    });
    if (error) {
      return res.status(500).json({ error: error.message });
    }
    return res.status(200).json({ message: 'Memory fact upserted' });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
});

app.post('/search_memory_facts', async (req, res) => {
  const { query, top_k } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'query is required' });
  }
  try {
    const queryEmbedding = await embedText(query);
    const { data, error } = await supabase.rpc('match_memory_facts', {
      query_embedding: queryEmbedding,
      match_count: top_k || 5,
      user_uuid: memoryOwnerUUID,
    });
    if (error) {
      return res.status(500).json({ error: error.message });
    }
    return res.status(200).json({ results: data });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
});

app.get('/', (req, res) => {
  res.json({ message: 'Open Brain Memory Server' });
});

app.listen(port, () => {
  console.log('Memory server listening on port ' + port);
});
