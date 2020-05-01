# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

def get_score1(args):
    cof = [1, 1]
    best_cof = [1]
    all_scores = collections.OrderedDict()
    idx = 0
    for input_file in args.input_null_files.split(","):
        with open(input_file, 'r') as reader:
            input_data = json.load(reader, strict=False)
            for (key, score) in input_data.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(cof[idx] * score)
        idx += 1
    output_scores = {}
    for (key, scores) in all_scores.items():
        mean_score = 0.0
        for score in scores:
            mean_score += score
        mean_score /= float(len(scores))
        output_scores[key] = mean_score

    idx = 0
    all_nbest = collections.OrderedDict()
    for input_file in args.input_nbest_files.split(","):
        with open(input_file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, entries) in input_data.items():
                if key not in all_nbest:
                    all_nbest[key] = collections.defaultdict(float)
                for entry in entries:
                    all_nbest[key][entry["text"]] += best_cof[idx] * entry["probability"]
        idx += 1
    output_predictions = {}
    for (key, entry_map) in all_nbest.items():
        sorted_texts = sorted(
            entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
        best_text = sorted_texts[0]
        output_predictions[key] = best_text

    best_th = args.thresh

    for qid in output_predictions.keys():
        if output_scores[qid] > best_th:
            output_predictions[qid] = ""

    output_prediction_file = "predictions.json"
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_predictions, indent=4) + "\n")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--input_null_files', type=str, default="cls_score.json,null_odds.json")
    parser.add_argument('--input_nbest_files', type=str, default="nbest_predictions.json")
    parser.add_argument('--thresh', default=0, type=float)
    parser.add_argument("--predict_file", default="data/dev-v2.0.json")
    args = parser.parse_args()
    get_score1(args)

if __name__ == "__main__":
    main()
