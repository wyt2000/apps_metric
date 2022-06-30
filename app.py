import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("loubnabnl/apps-metric")
launch_gradio_widget(module)