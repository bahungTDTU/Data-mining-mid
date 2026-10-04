"""Synthetic fixtures here only; research reports read the downloaded UCSD files."""
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import json
from scripts.survey_data import read_records, source_id, posted_date, eligibility
from scripts.survey_data import survey


class SurveyTests(unittest.TestCase):
    def test_literal_parser_rejects_executable_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as handle:
                handle.write("__import__('os').getcwd()\n")
            with self.assertRaisesRegex(ValueError, 'invalid literal'):
                list(read_records(path))

    def test_real_format_booleans_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.gz'
            expected = {'reviews': [{'recommend': True}], 'title': 'Trò chơi'}
            with gzip.open(path, 'wt', encoding='utf-8') as handle:
                handle.write(repr(expected) + '\n')
            self.assertEqual(list(read_records(path)), [expected])

    def test_ids_do_not_join_on_titles_or_null(self):
        self.assertEqual(source_id(10), '10')
        for value in (None, True, '', 'Portal', 1.5):
            self.assertIsNone(source_id(value))

    def test_missing_year_is_not_invented(self):
        self.assertEqual(posted_date('Posted November 5, 2011.'), '2011-11-05')
        self.assertIsNone(posted_date('Posted November 5.'))
        self.assertIsNone(posted_date('Posted February 30, 2011.'))

    def test_eligibility_requires_independent_target(self):
        self.assertEqual(eligibility({'a': [1], 'b': [1, 2], 'c': list(range(6))}), {'1': 2, '3': 1, '5': 1})

    def test_clean_join_conflicts_and_empty_users(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = [{'id': '1', 'title': 'One'}, {'id': '1', 'title': 'Duplicate'}, {'id': '2', 'title': 'Two'}]
            reviews = [
                {'user_id': 'a', 'reviews': [{'item_id': '1', 'recommend': True}, {'item_id': '1', 'recommend': True}, {'item_id': '9', 'recommend': True}]},
                {'user_id': 'a', 'reviews': [{'item_id': '2', 'recommend': True}, {'item_id': '2', 'recommend': False}]},
                {'user_id': 'b', 'reviews': []},
            ]
            for name, rows in [('metadata.gz', metadata), ('reviews.gz', reviews)]:
                with gzip.open(root / name, 'wt', encoding='utf-8') as handle:
                    for row in rows:
                        handle.write(repr(row) + '\n')
            config = {'schema_version': 'test-only', 'metadata': 'metadata.gz', 'reviews': 'reviews.gz', 'sample_users': 2, 'seed': 42, 'output': 'audit', 'processed': 'processed'}
            with patch('scripts.survey_data.ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                report = survey(config)
            self.assertEqual(report['counts']['conflicting_pairs_removed'], 1)
            self.assertEqual(report['counts']['duplicate_pair_rows'], 2)
            self.assertEqual(report['counts']['positive_unique_pairs_without_metadata'], 1)
            self.assertEqual(report['sample']['interactions'], 1)
            self.assertEqual(report['sample']['users'], 2)
            self.assertEqual(report['user_history']['histogram'], {0: 1, 1: 1})
            saved = json.loads((root / 'processed/games.json').read_text(encoding='utf-8'))
            self.assertEqual(saved[0]['title'], 'One')


if __name__ == '__main__':
    unittest.main()
