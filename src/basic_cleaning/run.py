#!/usr/bin/env python
"""Download a W&B artifact, perform basic cleaning, and log the result."""

import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact.
    logger.info("Downloading input artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info("Loading %s", artifact_local_path)
    df = pd.read_csv(artifact_local_path)
    logger.info("Loaded %d rows", len(df))

    # Drop price outliers (bounds are inclusive). Missing values are NOT imputed here:
    # imputation is done in the inference pipeline so it also applies at prediction time.
    logger.info("Filtering price outliers outside [%s, %s]", args.min_price, args.max_price)
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info("%d rows remain after price filtering", len(df))

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Drop listings outside the NYC area (found necessary with sample2.csv)
    # logger.info("Filtering listings outside the NYC geographical boundaries")
    # idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    # df = df[idx].copy()
    # logger.info("%d rows remain after geographical filtering", len(df))

    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    logger.info("Uploading %s to Weights & Biases as artifact %s", filename, args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)
    artifact.wait()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name and version of the raw input artifact in W&B (e.g. sample.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output artifact to create in W&B",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (e.g. clean_sample)",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Short description of the output artifact",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum accepted nightly price (inclusive); cheaper listings are dropped",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum accepted nightly price (inclusive); pricier listings are dropped",
        required=True,
    )

    args = parser.parse_args()
    go(args)
