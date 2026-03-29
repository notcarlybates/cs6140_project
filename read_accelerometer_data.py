"""
=========================================
Sample code to read the accelerometer data and labels into a single dataframe.
=========================================
Authors: Hoan Tran and Umberto Mazzucchelli
Email: tran[dot]hoan1[at]northeastern[dot]edu (train.hoan1@northeastern.edu)
"""
"""
=========================================
Sample code to read the accelerometer data and labels into a single dataframe.
Polars version for faster processing.
=========================================
"""

import sys
import polars as pl
from datetime import datetime, timedelta


def parse_header(file: str) -> tuple[datetime, int]:
    """Parse the actigraph file header to get start time and sampling rate."""
    sampling_rate = 1
    
    with open(file) as f:
        line = f.readline()
        parsed = line.split()
        
        for i in range(len(parsed)):
            if parsed[i] == "Hz":
                sampling_rate = int(parsed[i - 1])
                break
        
        f.readline()
        start_time = f.readline().split()[-1]
        start_date = f.readline().split()[-1]
    
    start = datetime.strptime(start_date + " " + start_time, "%m/%d/%Y %H:%M:%S")
    return start, sampling_rate


def read_data(file: str, agd: bool = False) -> pl.DataFrame:
    """Read actigraph data and add timestamps."""
    start, sampling_rate = parse_header(file)
    
    if agd:
        step_us = 1_000_000  # 1 second in microseconds
    else:
        step_us = int(1_000_000 / sampling_rate)
    
    df = pl.read_csv(file, skip_rows=10)
    
    # Add timestamps using Polars expressions (much faster than list comprehension)
    df = df.with_row_index("_idx").with_columns(
        (pl.lit(start).cast(pl.Datetime("us")) + pl.col("_idx") * timedelta(microseconds=step_us)).alias("Timestamp")
    ).drop("_idx")
    
    return df


def add_labels(accel: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Add activity labels using Polars join_asof and expressions."""
    
    data_start = labels["START_TIME"][0]
    data_end = labels["STOP_TIME"][-1]
    
    # Sort both dataframes by time (required for join_asof)
    accel = accel.sort("Timestamp")
    labels = labels.sort("START_TIME")
    
    # Join on START_TIME, then filter where Timestamp <= STOP_TIME
    result = accel.join_asof(
        labels.select(["START_TIME", "STOP_TIME", "ACTIVITY_CLASS"]),
        left_on="Timestamp",
        right_on="START_TIME",
        strategy="backward"
    )
    
    # Fix labels: set to null where Timestamp > STOP_TIME
    result = result.with_columns(
        pl.when(pl.col("Timestamp") > pl.col("STOP_TIME"))
        .then(pl.lit(None))
        .otherwise(pl.col("ACTIVITY_CLASS"))
        .alias("Activity")
    ).drop(["START_TIME", "STOP_TIME", "ACTIVITY_CLASS"])
    
    # Mark before/after data collection
    result = result.with_columns(
        pl.when(pl.col("Timestamp") < data_start)
        .then(pl.lit("Before_Data_Collection"))
        .when(pl.col("Timestamp") > data_end)
        .then(pl.lit("After_Data_Collection"))
        .otherwise(pl.col("Activity"))
        .alias("Activity")
    )
    
    return result


# Mapping scheme (inline to avoid utils dependency)
MAPPING_SCHEMES = {
    # Mapping with 5 activities, can be used with SimFL+Lab or FL data.
    "lab_fl_5": {
        "Sitting_Still": "Sitting",
        "Sitting_With_Movement": "Sitting",
        "Sit_Recline_Talk_Lab": "Sitting",
        "Sit_Recline_Web_Browse_Lab": "Sitting",
        "Sit_Writing_Lab": "Sitting",
        "Sit_Typing_Lab": "Sitting",

        "Standing_Still": "Standing",
        "Standing_With_Movement": "Standing",
        "Stand_Conversation_Lab": "Standing",

        "Lying_Still": "Lying_Down",
        "Lying_With_Movement": "Lying_Down",
        "Lying_On_Back_Lab": "Lying_Down",
        "Lying_On_Right_Side_Lab": "Lying_Down",
        "Lying_On_Stomach_Lab": "Lying_Down",
        "Lying_On_Left_Side_Lab": "Lying_Down",

        "Walking": "Walking",
        "Treadmill_2mph_Lab": "Walking",
        "Treadmill_3mph_Conversation_Lab": "Walking",
        "Treadmill_3mph_Free_Walk_Lab": "Walking",
        "Treadmill_3mph_Drink_Lab": "Walking",
        "Treadmill_3mph_Briefcase_Lab": "Walking",
        "Treadmill_3mph_Phone_Lab": "Walking",
        "Treadmill_3mph_Hands_Pockets_Lab": "Walking",
        "Walking_Fast": "Walking",
        "Walking_Slow": "Walking",

        'Stationary_Biking_300_Lab': 'Biking',
        'Exercising_Gym_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Regular_Bicycle': 'Biking',
    },

    # Mapping with 9 activities, can be used with SimFL+Lab or FL data.
    "lab_fl_9": {
        "Sitting_Still": "Sitting",
        "Sitting_With_Movement": "Sitting",
        "Sit_Recline_Talk_Lab": "Sitting",
        "Sit_Recline_Web_Browse_Lab": "Sitting",
        "Sit_Writing_Lab": "Sitting",
        "Sit_Typing_Lab": "Sitting",

        "Standing_Still": "Standing",
        "Standing_With_Movement": "Standing",
        "Stand_Conversation_Lab": "Standing",

        "Lying_Still": "Lying_Down",
        "Lying_With_Movement": "Lying_Down",
        "Lying_On_Back_Lab": "Lying_Down",
        "Lying_On_Right_Side_Lab": "Lying_Down",
        "Lying_On_Stomach_Lab": "Lying_Down",
        "Lying_On_Left_Side_Lab": "Lying_Down",

        "Walking": "Walking",
        "Treadmill_2mph_Lab": "Walking",
        "Treadmill_3mph_Conversation_Lab": "Walking",
        "Treadmill_3mph_Free_Walk_Lab": "Walking",
        "Treadmill_3mph_Drink_Lab": "Walking",
        "Treadmill_3mph_Briefcase_Lab": "Walking",
        "Treadmill_3mph_Phone_Lab": "Walking",
        "Treadmill_3mph_Hands_Pockets_Lab": "Walking",
        "Walking_Fast": "Walking",
        "Walking_Slow": "Walking",

        "Walking_Up_Stairs": "Walking_Up_Stairs",

        "Walking_Down_Stairs": "Walking_Down_Stairs",

        "Ab_Crunches_Lab": "Gym_Exercises",
        "Arm_Curls_Lab": "Gym_Exercises",
        "Push_Up_Lab": "Gym_Exercises",
        "Push_Up_Modified_Lab": "Gym_Exercises",
        "Machine_Leg_Press_Lab": "Gym_Exercises",
        "Machine_Chest_Press_Lab": "Gym_Exercises",
        "Treadmill_5.5mph_Lab": "Gym_Exercises",

        "Cycling_Active_Pedaling_Regular_Bicycle": "Biking",
        "Stationary_Biking_300_Lab": "Biking",

        'Organizing_Shelf/Closet': 'Household_Chores',
        'Sweeping': 'Household_Chores',
        'Vacuuming': 'Household_Chores',
        'Stand_Shelf_Load_Lab': 'Household_Chores',
        'Stand_Shelf_Unload_Lab': 'Household_Chores',
        'Washing_Dishes_Lab': 'Household_Chores',
        'Chopping_Food_Lab': 'Household_Chores'
    },

    # Mapping with 42 activities, used with SimFL+Lab data only.
    "lab_42": {
        "Ab_Crunches_Lab": "Ab_Crunches_Lab",
        "Arm_Curls_Lab": "Arm_Curls_Lab",
        "Chopping_Food_Lab": "Chopping_Food_Lab",
        "Cycling_Active_Pedaling_Regular_Bicycle":
            "Cycling_Active_Pedaling_Regular_Bicycle",
        "Folding_Clothes": "Folding_Clothes",
        "Lying_On_Back_Lab": "Lying_On_Back_Lab",
        "Lying_On_Left_Side_Lab": "Lying_On_Left_Side_Lab",
        "Lying_On_Right_Side_Lab": "Lying_On_Right_Side_Lab",
        "Lying_On_Stomach_Lab": "Lying_On_Stomach_Lab",
        "Machine_Chest_Press_Lab": "Machine_Chest_Press_Lab",
        "Machine_Leg_Press_Lab": "Machine_Leg_Press_Lab",
        "Organizing_Shelf/Cabinet": "Organizing_Shelf/Cabinet",
        "Playing_Frisbee": "Playing_Frisbee",
        "Push_Up_Lab": "Push_Up_Lab",
        "Push_Up_Modified_Lab": "Push_Up_Modified_Lab",
        "Sit_Recline_Talk_Lab": "Sit_Recline_Talk_Lab",
        "Sit_Recline_Web_Browse_Lab": "Sit_Recline_Web_Browse_Lab",
        "Sit_Typing_Lab": "Sit_Typing_Lab",
        "Sit_Writing_Lab": "Sit_Writing_Lab",
        "Sitting_Still": "Sitting_Still",
        "Sitting_With_Movement": "Sitting_With_Movement",
        "Stand_Conversation_Lab": "Stand_Conversation_Lab",
        "Stand_Shelf_Load_Lab": "Stand_Shelf_Load_Lab",
        "Stand_Shelf_Unload_Lab": "Stand_Shelf_Unload_Lab",
        "Standing_Still": "Standing_Still",
        "Standing_With_Movement": "Standing_With_Movement",
        "Stationary_Biking_300_Lab": "Stationary_Biking_300_Lab",
        "Sweeping": "Sweeping",
        "Treadmill_2mph_Lab": "Treadmill_2mph_Lab",
        "Treadmill_3mph_Conversation_Lab": "Treadmill_3mph_Conversation_Lab",
        "Treadmill_3mph_Drink_Lab": "Treadmill_3mph_Drink_Lab",
        "Treadmill_3mph_Free_Walk_Lab": "Treadmill_3mph_Free_Walk_Lab",
        "Treadmill_3mph_Hands_Pockets_Lab": "Treadmill_3mph_Hands_Pockets_Lab",
        "Treadmill_3mph_Briefcase_Lab": "Treadmill_3mph_Briefcase_Lab",
        "Treadmill_3mph_Phone_Lab": "Treadmill_3mph_Phone_Lab",
        "Treadmill_4mph_Lab": "Treadmill_4mph_Lab",
        "Treadmill_5.5mph_Lab": "Treadmill_5.5mph_Lab",
        "Vacuuming": "Vacuuming",
        "Walking": "Walking",
        "Walking_Down_Stairs": "Walking_Down_Stairs",
        "Walking_Up_Stairs": "Walking_Up_Stairs",
        "Washing_Dishes_Lab": "Washing_Dishes_Lab",
    },

    # Mapping with 11 activities, used for FL data only.
    "fl_11": {
        'Standing_Still': 'Standing',
        'Stand_Conversation_Lab': 'Standing',
        'Standing_With_Movement': 'Standing',

        'Sitting_Still': 'Sitting',
        'Sit_Typing_Lab': 'Sitting',
        'Sit_Recline_Web_Browse_Lab': 'Sitting',
        'Sit_Writing_Lab': 'Sitting',
        'Sit_Recline_Talk_Lab': 'Sitting',
        'Sitting_With_Movement': 'Sitting',

        'Lying_Still': 'Lying_Down',
        'Lying_On_Left_Side_Lab': 'Lying_Down',
        'Lying_On_Right_Side_Lab': 'Lying_Down',
        'Lying_On_Back_Lab': 'Lying_Down',
        'Lying_On_Stomach_Lab': 'Lying_Down',
        'Lying_With_Movement': 'Lying_Down',

        'Walking': 'Walking',
        'Walking_Slow': 'Walking',
        'Walking_Fast': 'Walking',
        'Walking_Up_Stairs': 'Walking',
        'Walking_Down_Stairs': 'Walking',
        'Walking_Treadmill': 'Walking',
        'Treadmill_2mph_Lab': 'Walking',
        'Treadmill_3mph_Conversation_Lab': 'Walking',
        'Treadmill_3mph_Drink_Lab': 'Walking',
        'Treadmill_3mph_Free_Walk_Lab': 'Walking',
        'Treadmill_3mph_Briefcase_Lab': 'Walking',
        'Treadmill_3mph_Hands_Pockets_Lab': 'Walking',
        'Treadmill_4mph_Lab': 'Walking',

        'Stationary_Biking_300_Lab': 'Biking',
        'Exercising_Gym_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Regular_Bicycle': 'Biking',

        'In_Transit_Driving_Car': 'Driving',

        'Playing_Frisbee': 'Exercising',
        'Playing_Exergame': 'Exercising',
        'Playing_Sports/Games': 'Exercising',
        'Doing_Resistance_Training_Free_Weights': 'Exercising',
        'Exercising_Gym_Other': 'Exercising',
        'Doing_Resistance_Training_Other': 'Exercising',
        'Exercising_Gym_Treadmill': 'Exercising',
        'Arm_Curls_Lab': 'Exercising',
        'Doing_Martial_Arts': 'Exercising',
        'Push_Up_Modified_Lab': 'Exercising',
        'Doing_Resistance_Training': 'Exercising',
        'Push_Up_Lab': 'Exercising',
        'Ab_Crunches_Lab': 'Exercising',
        'Machine_Leg_Press_Lab': 'Exercising',
        'Machine_Chest_Press_Lab': 'Exercising',
        'Running_Non_Treadmill': 'Exercising',
        'Running_Treadmill': 'Exercising',

        'Dry_Mopping': 'Household_Chores',
        'Cleaning': 'Household_Chores',
        'Dusting': 'Household_Chores',
        'Doing_Common_Housework_Light': 'Household_Chores',
        'Folding_Clothes': 'Household_Chores',
        'Doing_Common_Housework': 'Household_Chores',
        'Ironing': 'Household_Chores',
        'Doing_Dishes': 'Household_Chores',
        'Loading/Unloading_Washing_Machine/Dryer': 'Household_Chores',
        'Doing_Home_Repair_Light': 'Household_Chores',
        'Organizing_Shelf/Closet': 'Household_Chores',
        'Doing_Home_Repair': 'Household_Chores',
        'Putting_Clothes_Away': 'Household_Chores',
        'Packing/Unpacking_Something': 'Household_Chores',
        'Sweeping': 'Household_Chores',
        'Vacuuming': 'Household_Chores',
        'Stand_Shelf_Load_Lab': 'Household_Chores',
        'Stand_Shelf_Unload_Lab': 'Household_Chores',
        'Washing_Dishes_Lab': 'Household_Chores',

        'Chopping_Food_Lab': 'Cooking',
        'Cooking/Prepping_Food': 'Cooking',

        'Eating/Dining': 'Eating/Drinking',

        'Washing_Hands': 'Grooming',
        'Brushing_Teeth': 'Grooming',
        'Brushing/Combing/Tying_Hair': 'Grooming',
        'Washing_Face': 'Grooming',
        'Applying_Makeup': 'Grooming',
        'Flossing_Teeth': 'Grooming',
        'Blowdrying_Hair': 'Grooming',
    },
}

def data_to_csv(actigraph_path: str, label_path: str, output_path: str) -> None:
    """Combine actigraph data with labels and save to CSV."""
    
    print(f"Reading accel: {actigraph_path}")
    accel = read_data(actigraph_path)
    print(f"  Loaded {len(accel):,} rows")
    
    print(f"Reading labels: {label_path}")
    labels = pl.read_csv(label_path, try_parse_dates=True)
    
    # Map activity types to classes
    mapping = MAPPING_SCHEMES["lab_fl_5"]
    labels = labels.with_columns(
        pl.col("PA_TYPE").replace(mapping).alias("ACTIVITY_CLASS")
    )
    print(f"  Loaded {len(labels)} label rows")
    
    print("Merging...")
    result = add_labels(accel, labels)
    
    print(f"Writing: {output_path}")
    result.write_csv(output_path)
    print(f"  Done. {len(result):,} rows saved.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python read_accelerometer_data_polars.py <actigraph_path> <label_path> <output_path>")
        sys.exit(1)
    
    actigraph_path = sys.argv[1]
    label_path = sys.argv[2]
    output_path = sys.argv[3]
    
    data_to_csv(actigraph_path, label_path, output_path)