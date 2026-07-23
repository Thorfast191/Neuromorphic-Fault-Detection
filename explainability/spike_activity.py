"""
Spike telemetry collection for interpretability and sparsity
analysis of a trained LIFClassifier.
"""

from __future__ import annotations

import torch

from encoding.visualization import SpikeVisualizer
from evaluation.energy import sparsity


class SpikeActivityAnalyzer:
    """
    Parameters
    ----------
    model : models.networks.lif_classifier.LIFClassifier
    device : torch.device
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def collect(self, loader, max_batches: int | None = None) -> dict:
        """
        Run the model with `return_all=True` and gather per-batch
        input, hidden, and output spikes along with true labels.
        """

        self.model.eval()

        inputs_batches = []
        hidden_spikes_batches = []
        output_spikes_batches = []
        labels_batches = []

        with torch.no_grad():

            for i, (x, y) in enumerate(loader):

                if max_batches is not None and i >= max_batches:
                    break

                x = x.to(self.device)

                output = self.model(x, return_all=True)

                inputs_batches.append(x.cpu())
                hidden_spikes_batches.append(
                    [h.cpu() for h in output["hidden_spikes"]]
                )
                output_spikes_batches.append(output["spikes"].cpu())
                labels_batches.append(y)

        return {
            "inputs": inputs_batches,
            "hidden_spikes": hidden_spikes_batches,
            "output_spikes": output_spikes_batches,
            "labels": labels_batches,
        }

    def layer_sparsity(self, records: dict) -> dict[str, float]:
        """
        Average per-layer firing sparsity across all collected
        batches. Feeds directly into `evaluation.energy` comparisons.
        """

        n_hidden = len(records["hidden_spikes"][0])

        result = {}

        for layer_idx in range(n_hidden):

            layer_batches = [
                batch[layer_idx] for batch in records["hidden_spikes"]
            ]

            all_spikes = torch.cat(layer_batches, dim=1)  # (T, sum_B, size)

            result[f"hidden_{layer_idx}"] = sparsity(all_spikes)

        output_all = torch.cat(records["output_spikes"], dim=1)
        result["output"] = sparsity(output_all)

        return result

    def class_spike_profile(
        self,
        records: dict,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Mean output-neuron spike-count profile per true class - an
        interpretability check for whether the network learned
        separable per-class output representations.

        Returns
        -------
        Tensor, shape (num_classes, num_classes).
        Row i = average output spike-count vector for true class i.
        """

        counts_by_class = torch.zeros(num_classes, num_classes)
        samples_by_class = torch.zeros(num_classes)

        for output_spikes, labels in zip(
            records["output_spikes"], records["labels"]
        ):

            spike_counts = output_spikes.sum(dim=0)  # (B, num_classes)

            for label, count in zip(labels.tolist(), spike_counts):
                counts_by_class[label] += count
                samples_by_class[label] += 1

        samples_by_class = samples_by_class.clamp(min=1).unsqueeze(1)

        return counts_by_class / samples_by_class

    # ------------------------------------------------------------
    # Plotting - delegates to the existing SpikeVisualizer.
    # ------------------------------------------------------------

    def plot_sample_raster(
        self,
        records: dict,
        batch_index: int = 0,
        sample_index: int = 0,
        layer: str = "output",
    ):
        """
        Raster plot of one sample's spikes for a given layer
        ("output" or "hidden_<i>").
        """

        if layer == "output":
            spikes = records["output_spikes"][batch_index][:, sample_index, :]
        else:
            layer_idx = int(layer.split("_")[1])
            spikes = records["hidden_spikes"][batch_index][layer_idx][
                :, sample_index, :
            ]

        return SpikeVisualizer.raster(spikes.numpy())
