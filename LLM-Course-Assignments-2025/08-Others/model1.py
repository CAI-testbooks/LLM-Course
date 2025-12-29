from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# Transformer Encoder Layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

    # Feed Forward Part
    outputs = Dense(ff_dim, activation="relu")(attention)
    outputs = Dense(inputs.shape[-1])(outputs)
    outputs = Dropout(dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

    return outputs


# Function to create the model branch for 1-mer features with Transformer
def ourmodel_1mer(input_shape=(100, 150), head_size=128, num_heads=4, ff_dim=512, dropout=0.2, num_layers=3):
    inputs = Input(shape=input_shape)
    x = Convolution1D(128, 3, activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 3, activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(256, 3, activation='relu', padding='valid')(x)
    x = MaxPooling1D(pool_size=3, strides=1, padding='valid')(x)
    x = Dropout(dropout)(x)

    # Apply Transformer layers in a loop
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    return Model(inputs=inputs, outputs=x)


# Function to create the model branch for 4-mer features with Transformer
def ourmodel_4mer(input_shape=(97, 150), head_size=100, num_heads=4, ff_dim=512, dropout=0.2, num_layers=3):
    inputs = Input(shape=input_shape)

    # Apply Transformer layers in a loop
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = Flatten()(x)
    return Model(inputs=inputs, outputs=x)


# Define the combined model
def build_combined_model(model_1mer, model_4mer):
    # Inputs
    input_1mer = model_1mer.input
    input_4mer = model_4mer.input

    # Outputs from each branch
    output_1mer = model_1mer(input_1mer)
    output_4mer = model_4mer(input_4mer)

    # Concatenate the outputs along the feature axis
    combined_output = concatenate([output_1mer, output_4mer], axis=-1)

    # Add additional layers after concatenation if needed
    combined_output = Dense(64, activation='relu')(combined_output)
    combined_output = Dropout(0.5)(combined_output)
    final_output = Dense(2, activation='softmax')(combined_output)

    # Create the combined model
    combined_model = Model(inputs=[input_1mer, input_4mer], outputs=final_output)

    # Compile the model
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return combined_model


# Create instances of both models
model_1mer_instance = ourmodel_1mer()
model_4mer_instance = ourmodel_4mer()

# Create the combined model by passing the model instances
combined_model = build_combined_model(model_1mer_instance, model_4mer_instance)

# Print the model summary to verify the architecture
combined_model.summary()

