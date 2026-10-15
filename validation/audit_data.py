"""Reproduce structural diagnostics, not predictive performance. Run from repo root."""
import collections
import csv
import hashlib
import json
from pathlib import Path

root = Path('.')
with (root / 'Kaggle_train .csv').open() as f:
    train = list(csv.DictReader(f))
with (root / 'Kaggle_test.csv').open() as f:
    test = list(csv.DictReader(f))
groups = collections.Counter(row['Name'].split(',')[0] for row in train)
result = {
    'training_n': len(train),
    'test_n': len(test),
    'missing_training_age': sum(not row['Age'] for row in train),
    'surname_singleton_passengers': sum(n == 1 for n in groups.values()),
    'passengers_with_training_surname_peer': sum(n for n in groups.values() if n > 1),
    'original_script_sha256': hashlib.sha256(
        (root / 'Adaboost_Dec_27_leave_one_out.R').read_bytes()).hexdigest(),
}
assert not ({row['PassengerId'] for row in train} & {row['PassengerId'] for row in test})
assert len({row['PassengerId'] for row in train}) == len(train)
assert result['original_script_sha256'] == '70850780ff4e773e005f381dc50a14f378f91b9653bf1e47f5765d20a6131c7d'
(root / 'validation/feature_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
