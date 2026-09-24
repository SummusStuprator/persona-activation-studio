# Data and permissions

Use your own independently supplied writing, synthetic fixtures, or material for which collection, training and intended distribution are permitted. Identify real-person-derived personas as simulations. Their outputs are not that person's statements, memories, beliefs or endorsement.

Collection permission, author consent, platform terms and redistribution rights are separate. Reply parents/quoted posts can belong to others even when the target account is yours. Keep source/license/permission records privately with the dataset.

## X-specific restriction

Checked 24 September 2026: the [X Developer Agreement](https://docs.x.com/developer-terms/agreement), section III.A(k), prohibits using X API or X Content to train/fine-tune foundation or frontier models. The [Developer Policy](https://docs.x.com/developer-terms/policy) also addresses privacy, access and redistribution.

Do not assume public posts, author consent, an unofficial collector, an export or LoRA creates an exemption. Resolve the applicable terms/permission before using X-derived material; qualified advice may be needed for a particular agreement. Studio cannot certify that permission.

The collector uses [twscrape](https://github.com/vladkens/twscrape), an unofficial client. Authentication does not guarantee complete results or permitted downstream use. Respect blocks, protected/deleted content, rate limits and account restrictions.

## Local input format

Studio accepts its collector's CSV/JSONL format. It does not directly import a pasted profile URL, arbitrary text file or native X download archive. Convert permitted material to the schema first; do not invent parents and label them observed.

Point `[paths].exports` in `studio.toml` to:

```text
all_posts.csv
reply_context/
  parents.jsonl
```

`all_posts_with_context.csv` takes precedence if present. `parents.csv` can substitute for JSONL. An empty parent file is valid, but unresolved replies stay outside canonical supervised data. Optional `external_target_status.csv` records expected profiles.

| Field | Meaning |
|---|---|
| `id` | Unique string post ID |
| `username` | Author handle/profile |
| `date` | ISO timestamp, preferably consistent UTC |
| `kind` | `tweet`, `reply`, or quote metadata |
| `text` | Original authored text |
| `conversation_id` | Shared observed conversation ID |
| `in_reply_to_tweet_id` | Direct parent's ID, empty for standalone |
| `in_reply_to_username` | Parent author |
| `quoted_tweet_id` | Quoted source ID |
| `url` | Source URL, if any |
| `photos`, `videos`, `animated_gifs`, `links` | JSON arrays; `[]` when absent |

Parents use the same fields. An invented exchange illustrating the format:

```csv
id,username,date,kind,text,conversation_id,in_reply_to_tweet_id,in_reply_to_username
2,demo_author,2026-01-01T12:01:00Z,reply,I write down the steps first.,1,1,demo_reader
```

```json
{"id":"1","username":"demo_reader","date":"2026-01-01T12:00:00Z","kind":"tweet","text":"How do you start a difficult project?","conversation_id":"1"}
```

These two synthetic records are not enough to train. Use many independent conversations and inspect splits. `studio demo` provides a deterministic pipeline fixture.

## Retention and sharing

Keep exports, revisions, backups, credentials and adapters out of the source release. Renaming handles does not anonymize text/metadata. Review logs/traces before sharing.

Removal must account for exports, revisions, backups, caches, checkpoints and trained adapters. Deleting a CSV does not erase its learned influence; adapter retirement or retraining may be necessary.

Studio's MIT license does not grant rights to redistribute someone else's posts or model weights.
