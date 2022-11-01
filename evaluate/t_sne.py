from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10, viridis
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import Model
from io import BytesIO
import base64
import cv2
import numpy as np
import pandas as pd
from PIL import Image


def embeddable_image(data):
    image = cv2.resize(data, (64, 64), interpolation=cv2.INTER_CUBIC)

    # If the image is grayscale image, then uncomment the next line of code
    image = image[..., np.newaxis]

    image = array_to_img(image)
    buffer = BytesIO()
    image.save(buffer, format="png")
    for_encoding = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(for_encoding).decode()


def plot_tSNE(
    model,
    layername,
    input_data,
    input_label,
    input_predict,
    label_name,
    batch_size=64,
    modelname="Model",
):

    """
    Plot t-SNE (dimensionality reduction method)
    default: reduce to 2-dim

    [parameters]
        model: neural netowrk model

        layername: the layer we use t-SNE method to

        input_data: input data for the model (inputs=model.input, outputs=model.get_layer(layername).output)

        input_label: label of input data

        input_predict: predicted label of input data

        label_name: name of label

        batch_size: batch_size for model.predict

        modelname: name of model for saving

    """

    # dimensionality reduction
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layername).output
    )
    intermediate_output = intermediate_layer_model.predict(
        input_data, batch_size=batch_size, verbose=1
    )
    print("intermediate_output.shape:", intermediate_output.shape)

    t_SNE = TSNE(
        n_components=2, init="pca", random_state=0, perplexity=30, verbose=1
    ).fit_transform(intermediate_output)
    # print('t_SNE.shape:', t_SNE.shape)

    layer_output_label = np.argmax(input_label, axis=1)
    layer_output_label_predict = np.argmax(input_predict, axis=1)

    # bokeh for interactive visualization
    image = {}
    for num in range(len(label_name)):
        image[num] = []

    for index, true_label in enumerate(layer_output_label):
        for temp in range(len(label_name)):
            if temp == true_label:
                image[temp].append(index)
                break

    p = figure(
        title="t-SNE", plot_width=600, plot_height=600, tools=("pan, wheel_zoom, reset")
    )

    p.add_tools(
        HoverTool(
            tooltips="""
            <div>
                <div>
                    <img src='@image' height="22" width="100"
                        style='float: left; margin: 5px 5px 5px 5px'/>
                </div>
                <div>
                    <span style='font-size: 16px; color: #224499'>Label:</span>
                    <span style='font-size: 18px'>@label</span>
                </div>
                <div>
                    <span style='font-size: 16px; color: #224499'>Predict:</span>
                    <span style='font-size: 18px'>@predict</span>
                </div>
            </div>
        """
        )
    )

    df = []
    colors = viridis(len(label_name))
    for num in range(len(label_name)):
        df_temp = pd.DataFrame(t_SNE[image[num]], columns=("x", "y"))
        df_temp["label"] = [label_name[layer_output_label[x]] for x in image[num]]
        df_temp["predict"] = [
            label_name[layer_output_label_predict[x]] for x in image[num]
        ]
        df_temp["image"] = list(map(embeddable_image, input_data[image[num]]))

        datasource_temp = ColumnDataSource(df_temp)
        df.append(df_temp)

        p.circle(
            "x",
            "y",
            legend_label=label_name[num],
            source=datasource_temp,
            color=colors[num],
            muted_color=colors[num],
            muted_alpha=0.2,
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4,
        )

    p.legend.location = "bottom_left"

    # hide of mute mode
    p.legend.click_policy = "hide"

    output_file(modelname + " t-SNE.html")

    show(p)

    # df = [df_cat, df_dog]
    df = pd.concat(df, ignore_index=True)

    # Save tSNE Result
    # float_format: control precision
    writer = pd.ExcelWriter(modelname + " tSNE {}.xlsx".format(layername))
    df.to_excel(writer, sheet_name="Intensity", float_format="%.2f")
    writer.save()
