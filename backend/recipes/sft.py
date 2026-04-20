import chz
import sys
import os
from dotenv import load_dotenv  # noqa: E402

# Load environment variables from multiple possible locations
load_dotenv()  # Load from current directory
if "TINKER_ENV_FILE" in os.environ:
    load_dotenv(os.environ["TINKER_ENV_FILE"])  # Load from specified file

import datasets  # noqa: E402
from tinker_cookbook import cli_utils, model_info  # noqa: E402
from tinker_cookbook.renderers import TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised.data import (  # noqa: E402
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (  # noqa: E402
    ChatDatasetBuilderCommonConfig,
    ChatDatasetBuilder,
)
from tinker_cookbook.supervised import train  # noqa: E402
from tinker_cookbook.hyperparam_utils import get_lr  # noqa: E402
import asyncio  # noqa: E402


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    # Use a default model, but allow override from command line
    model_name = (
        "meta-llama/Llama-3.1-8B-Instruct"  # Default to Llama for compatibility
    )
    dataset_name = "yahma/alpaca-cleaned"  # Default dataset
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=128,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    @chz.chz
    class AlpacaBuilder(ChatDatasetBuilder):
        dataset: str = dataset_name

        def __call__(self) -> tuple:
            # Use the specified dataset
            ds = datasets.load_dataset(self.dataset)
            train_ds = (
                ds["train"].shuffle(seed=0).select(range(1000))
            )  # Limit to 1000 samples for quick testing
            # Use a subset of train data for testing if no test split exists
            if "test" in ds:
                test_ds = ds["test"].select(range(min(100, len(ds["test"]))))
            else:
                test_ds = ds["train"].select(
                    range(1000, 1100)
                )  # Use part of train for testing

            def map_fn(row: dict):
                instr = (row.get("instruction") or "").strip()
                inp = (row.get("input") or "").strip()
                if inp:
                    user_content = f"{instr}\n\nInput:\n{inp}"
                else:
                    user_content = instr
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": row.get("output", "")},
                ]
                return conversation_to_datum(
                    messages,
                    self.renderer,
                    common_config.max_length,
                    common_config.train_on_what,
                )

            return (
                SupervisedDatasetFromHFDataset(
                    train_ds, batch_size=common_config.batch_size, map_fn=map_fn
                ),
                SupervisedDatasetFromHFDataset(
                    test_ds, batch_size=common_config.batch_size, map_fn=map_fn
                ),
            )

    dataset_builder = AlpacaBuilder(common_config=common_config)

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/tinker-examples/sl_basic",
            "model_name": model_name,
            "dataset_builder": dataset_builder,
            "learning_rate": get_lr(model_name),
            "lr_schedule": "linear",
            "num_epochs": 1,
            "eval_every": 8,
            "wandb_project": None,
            "wandb_name": None,
        }
    )


def main(config: train.Config):
    # Verify Tinker API key is available (SECURITY: no logging of key values)
    if not os.getenv("TINKER_API_KEY"):
        print("ERROR: TINKER_API_KEY not found in environment variables")
        print("Please ensure TINKER_API_KEY is set in your .env file")
        sys.exit(1)

    print("TINKER_API_KEY configured successfully")

    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
