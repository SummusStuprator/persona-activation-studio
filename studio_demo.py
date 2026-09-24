"""Create a local arithmetic dataset and a tiny adapter for pipeline testing."""
from pathlib import Path
import argparse
import json
import uuid
from studio_config import ensure_workspace


def create_dataset(folder):
    split_root = folder / 'profiles/demo/sft/portable/core'
    split_root.mkdir(parents=True)
    for split, start, count in (('train', 10, 32), ('validation', 100, 8), ('test', 200, 8)):
        rows = []
        for number in range(start, start + count):
            rows.append({'id': f'{split}-{number}', 'profile': 'demo',
                         'messages': [{'role': 'user', 'content': f'Calculate {number} plus 7.'},
                                      {'role': 'assistant', 'content': f'The result is {number + 7}.'}],
                         'metadata': {'conversation_id': f'{split}-{number}', 'source': 'arithmetic_fixture'}})
        filename = 'train_authentic.jsonl' if split == 'train' else split + '.jsonl'
        (split_root / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return split_root


def train_fixture(folder, dataset, destination, device):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from persona.trainer_v4 import (load_filtered_splits, tokenize_persona_row,
        CapabilityPreservingTrainer, PersonaDataset, DynamicCollator)
    if device == 'CUDA' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    train, validation, test, _, _ = load_filtered_splits(dataset, {'profile': 'demo'})
    words = ['<unk>', '<bos>', '<eos>', '<pad>', 'user', 'assistant', ':', '.']
    words += ['Calculate', 'plus', 'The', 'result', 'is'] + [str(n) for n in range(220)]
    vocab = dict.fromkeys(words)
    vocab = {word: index for index, word in enumerate(vocab)}
    raw = Tokenizer(WordLevel(vocab, unk_token='<unk>')); raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='<unk>',
        bos_token='<bos>', eos_token='<eos>', pad_token='<pad>')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=3)
    torch.manual_seed(42)
    base = Qwen2ForCausalLM(config)
    base_path = folder / 'base'; base.save_pretrained(base_path); tokenizer.save_pretrained(base_path)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', lora_dropout=0.))
    model.peft_config['default'].base_model_name_or_path = str(base_path)
    features = [tokenize_persona_row(tokenizer, row.messages, 64, 24) for row in train]
    args = TrainingArguments(output_dir=str(folder / 'checkpoints'), max_steps=2,
        use_cpu=device == 'CPU', per_device_train_batch_size=1, learning_rate=.001,
        save_strategy='steps', save_steps=1, logging_strategy='no', report_to=[],
        remove_unused_columns=False, disable_tqdm=True, seed=42)
    trainer = CapabilityPreservingTrainer(model=model, args=args,
        train_dataset=PersonaDataset(features), data_collator=DynamicCollator(tokenizer))
    result = trainer.train()
    destination.mkdir(parents=True)
    model.save_pretrained(destination / 'adapter'); tokenizer.save_pretrained(destination / 'adapter')
    report = dict(profile='demo', fixture=True, base_model=str(base_path),
                  optimizer_steps=trainer.state.global_step, training_loss=result.training_loss)
    (destination / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = [{'id': r.row_id, 'profile': 'demo', 'messages': r.messages} for r in test]
    (destination / 'prepared_test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--device', choices=['CPU', 'CUDA'], default='CPU')
    args = parser.parse_args()
    paths = ensure_workspace(); identifier = uuid.uuid4().hex[:8]
    folder = paths.workspace / 'demo' / identifier
    folder.mkdir(parents=True)
    dataset = paths.datasets / ('demo-' + identifier)
    create_dataset(dataset)
    profile_path = paths.workspace / 'profiles.json'
    profiles = json.loads(profile_path.read_text()) if profile_path.exists() else {'profiles': []}
    if not any(p.get('profile') == 'demo' for p in profiles['profiles']):
        profiles['profiles'].append({'profile': 'demo', 'dataset_tier': 'core', 'max_steps': 2})
        profile_path.write_text(json.dumps(profiles, indent=2), encoding='utf-8')
    report = {'dataset': str(dataset), 'fixture': 'arithmetic', 'profile': 'demo'}
    if args.train:
        from workshop_v2.resource_policy import ModelLaunchLease, ModelLease, checkpoint
        destination = paths.models / ('demo-' + identifier)
        with ModelLaunchLease(), ModelLease():
            checkpoint()
            report.update(train_fixture(folder, dataset, destination, args.device))
        report['model'] = str(destination)
    (folder / 'demo.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
