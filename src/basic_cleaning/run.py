#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def _preprocess_data(df, min_price, max_price) -> pd.DataFrame:
    """Function to execute basic data preprocessing routines

    Args:
        df (pd.Dataframe): raw input dataset
        min_price (float): minimum room/apartment price
        max_price (float: maximum room/apartment price
    Returns:
        df (pd.Dataframe): Cleaned dataset
    """
    logger.info("[data-cleaning] - Preprocessing raw data")
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    df['last_review'] = pd.to_datetime(df['last_review'])
    return df


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################

    logger.info("[data-cleaning] - Loading artifact from W&B")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    cleaned_data = _preprocess_data(
        df=df,
        min_price=args.min_price,
        max_price=args.max_price
    )

    cleaned_data.to_csv("clean_sample.csv", index=False)

    logger.info("[data-cleaning] - Uploading cleaned data to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact with raw data",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact with clean data",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of output data artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum room/apartment price to remove outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum room/apartment price to remove outliers",
        required=True
    )

    args = parser.parse_args()

    go(args)
