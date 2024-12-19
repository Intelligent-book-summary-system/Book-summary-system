#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from rouge_score import rouge_scorer
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict


class RougeEvaluator:
    def __init__(self, metrics: List[str] = ['rouge1', 'rouge2', 'rougeL']):
        self.scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        self.metrics = metrics

    def evaluate_summary(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        scores = self.scorer.score(reference, candidate)
        return scores

    def load_references(self, reference_path: str) -> Dict:
        if reference_path.endswith('.json'):
            with open(reference_path, 'r', encoding='utf-8') as f:
                references = json.load(f)
        elif reference_path.endswith('.pkl'):
            with open(reference_path, 'rb') as f:
                references = pickle.load(f)
        else:
            raise ValueError("Unsupported file format for references. Use .json or .pkl")
        return references

    def load_summaries(self, summary_path: str) -> Dict:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
        return summaries

    def evaluate_book_summaries(self,
                                summary_path: str,
                                reference_path: str,
                                summary_type: str = 'hier') -> Tuple[Dict, pd.DataFrame]:
        # Load summaries and references
        summaries = self.load_summaries(summary_path)
        print(f"Loaded {len(summaries)} summaries from '{summary_path}'.")
        references = self.load_references(reference_path)
        print(f"Loaded {len(references)} references from '{reference_path}'.")

        results = defaultdict(list)
        detailed_scores = {}

        # Evaluate each book
        for book_id in summaries:
            if book_id not in references:
                print(f"Book ID '{book_id}' not found in references. Skipping.")
                continue

            # Get generated summary based on type
            if summary_type == 'hier':
                # Handle both dict and string summaries
                if isinstance(summaries[book_id], dict):
                    candidate = summaries[book_id].get('final_summary', '')
                    print(f"Book ID '{book_id}': Using 'final_summary'.")
                elif isinstance(summaries[book_id], str):
                    candidate = summaries[book_id]
                    print(f"Book ID '{book_id}': Using direct summary string.")
                else:
                    raise ValueError(f"Unsupported summary format for book {book_id}: {type(summaries[book_id])}")
            else:  # incremental
                candidate = summaries[book_id][-1] if isinstance(summaries[book_id], list) else ''
                print(f"Book ID '{book_id}': Using last incremental summary.")

            reference = references.get(book_id, '')
            if not reference:
                print(f"Book ID '{book_id}': Reference summary is empty. Skipping.")
                continue

            # Calculate ROUGE scores
            scores = self.evaluate_summary(reference, candidate)

            # Store detailed scores
            detailed_scores[book_id] = {
                'candidate': candidate,
                'reference': reference,
                'scores': scores
            }

            # Store summary metrics
            for metric in self.metrics:
                results[f"{metric}_precision"].append(scores[metric].precision)
                results[f"{metric}_recall"].append(scores[metric].recall)
                results[f"{metric}_fmeasure"].append(scores[metric].fmeasure)

        summary_df = pd.DataFrame(results)
        print("Evaluation completed.")
        return detailed_scores, summary_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate summaries using ROUGE metrics')

    parser.add_argument('--summaries', '-s',
                        required=True,
                        help='Path to the generated summaries JSON file')

    parser.add_argument('--references', '-r',
                        required=True,
                        help='Path to the reference summaries file (.json or .pkl)')

    parser.add_argument('--type', '-t',
                        choices=['hier', 'inc'],
                        default='hier',
                        help='Type of summarization (hierarchical or incremental)')

    parser.add_argument('--metrics', '-m',
                        nargs='+',
                        default=['rouge1', 'rouge2', 'rougeL'],
                        choices=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                        help='ROUGE metrics to compute')

    parser.add_argument('--output-dir', '-o',
                        default='rouge_results',
                        help='Directory to save evaluation results')

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print detailed results')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = RougeEvaluator(metrics=args.metrics)

    # Run evaluation
    print(f"\nEvaluating summaries using {', '.join(args.metrics)}...")
    try:
        detailed_scores, summary_df = evaluator.evaluate_book_summaries(
            args.summaries,
            args.references,
            args.type
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        exit(1)

    # Save results
    summary_df.to_csv(os.path.join(args.output_dir, 'rouge_scores.csv'))
    with open(os.path.join(args.output_dir, 'detailed_scores.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed_scores, f, indent=2)

    # Print results
    print("\nOverall ROUGE Scores:")
    print("\nMean scores:")
    print(summary_df.mean().round(4))
    print("\nStandard deviation:")
    print(summary_df.std().round(4))

    if args.verbose:
        print("\nDetailed scores by book:")
        for book_id, data in detailed_scores.items():
            print(f"\nBook: {book_id}")
            for metric in args.metrics:
                score = data['scores'].get(metric, None)
                if score:
                    print(f"{metric}:")
                    print(f"  Precision: {score.precision:.4f}")
                    print(f"  Recall: {score.recall:.4f}")
                    print(f"  F1: {score.fmeasure:.4f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
