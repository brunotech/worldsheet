# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from mmf.common.sample import Sample
from mmf.datasets.builders.vqa2 import VQA2Dataset


class VizWizDataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="vizwiz",
            *args,
            **kwargs
        )

    def load_item(self, idx):
        return super().load_item(idx)

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        answer_space_size = self.answer_processor.get_true_vocab_size()

        predictions = []
        for idx, image_id in enumerate(report.image_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
            else:
                answer = self.answer_processor.idx2word(answer_id)
            # if answer == self.context_processor.PAD_TOKEN:
            #     answer = "unanswerable"
            if answer in ["<unk>", "<pad>"]:
                answer = "unanswerable"
            predictions.append(
                {
                    "image": f"VizWiz_{self._dataset_type}_{str(image_id.item()).zfill(12)}.jpg",
                    "answer": answer,
                }
            )

        return predictions
