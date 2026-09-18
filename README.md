
# Protein Mutation Stability Prediction

This project estimates how a single amino-acid mutation may affect protein stability. Given a wild-type protein sequence and a mutation such as `V42A`, it predicts the expected change in stability as a continuous ddG value.

The project uses the public MegaScale protein-stability dataset. It validates sequences and mutation notation, keeps related proteins separated across training and evaluation splits, and trains a reproducible baseline and frozen-ESM regression head with documented held-out results.

The goal is to help researchers prioritize single mutations for experimental testing. Predictions are computational estimates and are not a replacement for laboratory measurements.


