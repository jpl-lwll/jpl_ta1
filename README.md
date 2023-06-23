# JPL TA1

This repo serves to demonstrate automated interaction with the API.

The secondary purpose of this is to test full coverage across every task with full and sample partitions in order to show working interactions.

## Running the System

To run the system, you can launch the `main.py` with the following options:

-   `dataset_type`
-   -   Can take on values of `sample`, `full`, or `all`
-   -   Determines which slice of the dataset we want to run against or if we want to do both for verifying complete integrity
-   `problem_type`
-   -   Can take on values of `image_classification`, `object_detection`, `video_classification`,
        `machine_translation`, or `all`
-   -   Determines which problem type we want to run against
-   `dataset_dir`
-   -   The directory that contains the `external` and `development` subdirectories while on a non evaluation environment or `external` and `evaluation` subdirectories while on an evaluation environment
-   -   Ex. `~/lwll_datasets`
-   `environment`
-   -   Can take on values of `local`, `dev`, `staging`, or `prod`
-   -   This controls validation on checking our `dataset_dir` subdirectories in addition to adjusting our internal api endpoint pointer to point to the appropriate endpoint. Note if you are using `local`, you must start the local api with all proper environment variables and credentials files locally on port 5000. Using in `local` mode is mostly only for developers of the `lwll_api` repository.
-   `team_secret`
-   `log_level`
-   -   Can take on values of `DEBUG`, `INFO`, `WARNING`, or `ERROR`
-   -   These log levels expose different amounts of logging to the terminal. These levels obey normal logging level standards.
-   `task_id`
-   -   Optional: allows you to run a single specific task
-   -   Ex. `6d5e1f85-5d8f-4cc9-8184-299db03713f4`

## Example Launches

Launch the system to test against only `sample` versions of `object_detection` problem types

```
python main.py launch_system --dataset_type sample --problem_type object_detection --dataset_dir ~/lwll_datasets --environment dev --team_secret $LWLL_SECRET --log_level DEBUG
```

Launch the system to test against `full` versions of the datasets for `all` problem types


```
python main.py launch_system --dataset_type full --problem_type all --dataset_dir ~/lwll_datasets --environment staging --team_secret $LWLL_SECRET --log_level DEBUG
```

Launch full diagnostics checking `sample` AND `full` across `all` problem types. This is used to allow us to test that there are no odd one off cases across all splits and tasks.

```
python main.py launch_system --dataset_type all --problem_type all --dataset_dir ~/lwll_datasets --environment staging --team_secret $LWLL_SECRET --log_level INFO
```