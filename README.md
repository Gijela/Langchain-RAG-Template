# langchain-RAG-Template

> 基于 [langchain-ai/langchain](https://github.com/langchain-ai/langchain-nextjs-template) 项目学习，原项目 [README](./README-EN.MD)。

已跑通 chat、RAG、RAG-Agents，成功引入使用的相关代码 [commit](https://github.com/Gijela/CR-Mentor/commit/1f71b6f69e708c7eb61fdc5a2ee42633042fbb11)

## 1. 用到的环境变量

```json
OPENAI_API_KEY=
OPENAI_API_BASE=
SUPABASE_PRIVATE_KEY=
SUPABASE_URL=
```

## 2. supabase 执行 SQL 创建匹配函数和表

1024 维向量对应的是 `BAAI/bge-large-zh-v1.5` 模型

```sql
-- 删除旧的表和索引
drop index if exists documents_embedding_idx;
drop table if exists documents;

-- 创建新的表，使用1024维向量
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb default '{}'::jsonb,
  embedding vector(1024)  -- 改为1024维
);

-- 更新匹配函数
create or replace function match_documents (
  query_embedding vector(1024),  -- 改为1024维
  filter jsonb default '{}'::jsonb,
  match_count int default 5,
  match_threshold float8 default 0.8
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float8
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where case
    when filter = '{}'::jsonb then true
    else documents.metadata @> filter
  end
  and 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 创建新的索引
create index documents_embedding_idx
  on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
```

## 3. curl 测试

### 3.1 RAG 嵌入&存储

接口：`/api/retrieval/ingest`

```bash
curl -X POST \
  http://localhost:3000/api/retrieval/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "这是一段测试文本。\n\n这是第二段测试文本。\n\n这是包含一些专业术语的第三段文本：人工智能、机器学习、深度学习等。"
  }'
```

```json
响应：{"ok":true}%
```

### 3.2 RAG 检索

接口：`/api/chat/retrieval`

```bash
curl -X POST http://localhost:3000/api/chat/retrieval \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ]
  }'
```

```json
响应：汪汪！我是Dana，一只充满活力的小狗狗！🐾 我最喜欢的事情就是追着自己的尾巴转圈圈，还有和主人一起玩捡球游戏！🏈 我的舌头总是伸在外面，因为我觉得这样更酷炫！😎 我还有一双超级灵敏的耳朵，能听到主人的每一个呼唤！👂 如果你有任何问题，尽管问我，我会用我那充满智慧的小脑袋瓜帮你解答！🧠 汪汪，让我们一起度过一个充满欢笑的时光吧！🎉%
```

### 3.3 RAG检索 + Agent

接口：`/api/chat/retrieval_agent`

```bash
curl -X POST http://localhost:3000/api/chat/retrieval_agents \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下自己"
      }
    ]
  }'
```

```json
响应：你好！我是机器人助手Robbie，BEEP BOOP！我是一个专门设计来帮助回答问题和提供信息的机器人。我的目标是尽可能地帮助你，无论是解答疑问还是提供最新的信息。如果你有任何问题，尽管问我，我会尽力帮助你！BOOP BEEP！%
```
