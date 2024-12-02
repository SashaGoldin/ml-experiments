import io
import keras
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.express as px

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Ran the import statements.")

# Load the dataset
rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

# Read and provide statistics on the dataset.
rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
]]

rice_dataset.describe()

# Create five 2D plots of the features against each other, color-coded by class.
for x_axis_data, y_axis_data in [
    ('Area', 'Eccentricity'),
    ('Convex_Area', 'Perimeter'),
    ('Major_Axis_Length', 'Minor_Axis_Length'),
    ('Perimeter', 'Extent'),
    ('Eccentricity', 'Major_Axis_Length'),
]:
    px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color='Class').show()

# Calculate the Z-scores of each numerical column in the raw data and write
# them into a new DataFrame named df_norm.

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (
                             rice_dataset[numerical_features] - feature_mean
                     ) / feature_std

# Copy the class to the new dataframe
normalized_dataset['Class'] = rice_dataset['Class']

# Examine some of the values of the normalized training set. Notice that most
# Z-scores fall between -2 and +2.
normalized_dataset.head()

# set the random seeds
keras.utils.set_random_seed(42)

# Create a column setting the Cammeo label to '1' and the Osmancik label to '0'
# then show 10 randomly selected rows.
normalized_dataset['Class_Bool'] = (
    # Returns true if class is Cammeo, and false if class is Osmancik
        normalized_dataset['Class'] == 'Cammeo'
).astype(int)
normalized_dataset.sample(10)

# Create indices at the 80th and 90th percentiles
number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

# Randomize order and split into train, validation, and test with a .8, .1, .1 split
shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

# Show the first five rows of the last split
test_data.head()

# prevent label lekeage (store features and labels in different variables)

label_columns = ['Class', 'Class_Bool']

train_features = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

# Name of the features we'll train our model on.
input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Area',
]

# @title Define the functions that create and train a model.

import dataclasses


@dataclasses.dataclass()
class ExperimentSettings:
    """Lists the hyperparameters and input features used to train am model."""

    learning_rate: float
    number_epochs: int
    batch_size: int
    classification_threshold: float
    input_features: list[str]


@dataclasses.dataclass()
class Experiment:
    """Stores the settings used for a training run and the resulting model."""

    name: str
    settings: ExperimentSettings
    model: keras.Model
    epochs: np.ndarray
    metrics_history: keras.callbacks.History

    def get_final_metric_value(self, metric_name: str) -> float:
        """Gets the final value of the given metric for this experiment."""
        if metric_name not in self.metrics_history:
            raise ValueError(
                f'Unknown metric {metric_name}: available metrics are'
                f' {list(self.metrics_history.columns)}'
            )
        return self.metrics_history[metric_name].iloc[-1]


def create_model(
        settings: ExperimentSettings,
        metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple classification model."""
    model_inputs = [
        keras.Input(name=feature, shape=(1,))
        for feature in settings.input_features
    ]
    # Use a Concatenate layer to assemble the different inputs into a single
    # tensor which will be given as input to the Dense layer.
    # For example: [input_1[0][0], input_2[0][0]]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    dense = keras.layers.Dense(
        units=1, input_shape=(1,), name='dense_layer', activation=keras.activations.sigmoid
    )
    model_output = dense(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)
    # Call the compile method to transform the layers into a model that
    # Keras can execute.  Notice that we're using a different loss
    # function for classification than for regression.
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            settings.learning_rate
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model


def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        labels: np.ndarray,
        settings: ExperimentSettings,
) -> Experiment:
    """Feed a dataset into the model in order to train it."""

    # The x parameter of keras.Model.fit can be a list of arrays, where
    # each array contains the data for one feature.
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )

    return Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


print('Defined the create_model and train_model functions.')


# Define the plotting function.
def plot_experiment_metrics(experiment: Experiment, metrics: list[str]):
    """Plot a curve of one or more metrics for different epochs."""
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        plt.plot(
            experiment.epochs, experiment.metrics_history[metric], label=metric
        )

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.grid()
    plt.legend()


print("Defined the plot_curve function.")

# Let's define our first experiment settings.
settings = ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name='precision', thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# Establish the model's topography.
model = create_model(settings, metrics)

# Train the model on the training set.
experiment = train_model(
    'baseline', model, train_features, train_labels, settings
)

# Plot metrics vs. epochs
plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
plot_experiment_metrics(experiment, ['auc'])


def evaluate_experiment(
        experiment: Experiment, test_dataset: pd.DataFrame, test_labels: np.array
) -> dict[str, float]:
    features = {
        feature_name: np.array(test_dataset[feature_name])
        for feature_name in experiment.settings.input_features
    }
    return experiment.model.evaluate(
        x=features,
        y=test_labels,
        batch_size=settings.batch_size,
        verbose=0,  # Hide progress bar
        return_dict=True,
    )


def compare_train_test(experiment: Experiment, test_metrics: dict[str, float]):
    print('Comparing metrics between train and test:')
    for metric, test_value in test_metrics.items():
        print('------')
        print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'Test {metric}:  {test_value:.4f}')


# Evaluate test metrics
test_metrics = evaluate_experiment(experiment, test_features, test_labels)
compare_train_test(experiment, test_metrics)


# Define function to compare experiments

def compare_experiment(experiments: list[Experiment],
                       metrics_of_interest: list[str],
                       test_dataset: pd.DataFrame,
                       test_labels: np.array):
    # Make sure that we have all the data we need.
    for metric in metrics_of_interest:
        for experiment in experiments:
            if metric not in experiment.metrics_history:
                raise ValueError(f'Metric {metric} not available for experiment {experiment.name}')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(2, 1, 1)

    colors = [f'C{i}' for i in range(len(experiments))]
    markers = ['.', '*', 'd', 's', 'p', 'x']
    marker_size = 10

    ax.set_title('Train metrics')
    for i, metric in enumerate(metrics_of_interest):
        for j, experiment in enumerate(experiments):
            plt.plot(experiment.epochs, experiment.metrics_history[metric], markevery=4,
                     marker=markers[i], markersize=marker_size, color=colors[j])

    # Add custom legend to show what the colors and markers mean
    legend_handles = []
    for i, metric in enumerate(metrics_of_interest):
        legend_handles.append(Line2D([0], [0], label=metric, marker=markers[i],
                                     markersize=marker_size, c='k'))
    for i, experiment in enumerate(experiments):
        legend_handles.append(Line2D([0], [0], label=experiment.name, color=colors[i]))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.grid()
    ax.legend(handles=legend_handles)

    ax = fig.add_subplot(2, 1, 2)
    spacing = 0.3
    n_bars = len(experiments)
    bar_width = (1 - spacing) / n_bars
    for i, experiment in enumerate(experiments):
        test_metrics = evaluate_experiment(experiment, test_dataset, test_labels)
        x = np.arange(len(metrics_of_interest)) + bar_width * (i + 1 / 2 - n_bars / 2)
        ax.bar(x, [test_metrics[metric] for metric in metrics_of_interest], width=bar_width, label=experiment.name)
    ax.set_xticks(np.arange(len(metrics_of_interest)), metrics_of_interest)

    ax.set_title('Test metrics')
    ax.set_ylabel('Metric value')
    ax.set_axisbelow(True)  # Put the grid behind the bars
    ax.grid()
    ax.legend()


print('Defined function to compare experiments.')
