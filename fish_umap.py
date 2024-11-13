import base64
import pandas as pd
from glob import glob
from io import BytesIO
from os.path import basename
from arrow import now
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from umap import UMAP
import plotly.express as express
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_notebook, show
from bokeh.palettes import Set1_3
from bokeh.transform import factor_cmap
import warnings

# Setup for warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Constants and Paths
SIZE = 512
STOP = 10000
DATA_GLOB = '/Users/akashadhyapak/Documents/ML/Fish Disease/Kaggle_upload'  # Update this path

# Flatten Function
def flatten(arg):
    return [x for xs in arg for x in xs]

# Function to load images and encode them into vectors
def get_from_glob(arg: str, tag: str, stop: int) -> list:
    time_get = now()
    result = []
    for index, input_file in enumerate(glob(pathname=arg, recursive=True)):
        if index < stop:
            if input_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Ensure only image files
                name = input_file.replace(DATA_GLOB, '')
                with Image.open(fp=input_file, mode='r') as image:
                    vector = img2vec.get_vec(image, tensor=True).numpy().reshape(SIZE,)
                    buffer = BytesIO()
                    size = (128, 128)
                    image.resize(size=size).save(buffer, format='png')
                    result.append(pd.Series(data=[tag, name, vector,
                                                  'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                                                 ], index=['tag', 'name', 'value', 'image']))
    print(f'encoded {tag} data {len(result)} rows in {now() - time_get}')
    return result

# Initialize Img2Vec with ResNet-18
img2vec = Img2Vec(cuda=False, model='resnet-18', layer='default', layer_output_size=SIZE)

# Process images and get vectors
time_start = now()

# Get all image files from Fresh and NonFresh folders
files = {basename(folder): folder + '/**/*.jpg' for folder in glob(DATA_GLOB + '/*')}

# Debug print statement to ensure correct files are picked
print("Files to process: ", files)

# Process data for all classes
data = [get_from_glob(arg=value, tag=key, stop=STOP) for key, value in files.items()]
df = pd.DataFrame(data=flatten(arg=data))
print('done in {}'.format(now() - time_start))

# Train-Test Split (use 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(df['value'].apply(func=pd.Series), df['tag'], test_size=0.3, random_state=63, stratify=df['tag'])

# Logistic Regression Model with stronger regularization to reduce overfitting
model = LogisticRegression(max_iter=100000, tol=1e-4, C=0.01).fit(X=X_train, y=y_train)
print('accuracy: {:5.4f}'.format(accuracy_score(y_true=y_test, y_pred=model.predict(X=X_test))))
print(classification_report(y_true=y_test, y_pred=model.predict(X=X_test)))

# UMAP Dimensionality Reduction
umap = UMAP(random_state=2024, verbose=True, n_jobs=1, low_memory=False, n_epochs=100)
df[['x', 'y']] = umap.fit_transform(X=df['value'].apply(func=pd.Series))

# Plot with Plotly
express.scatter(data_frame=df, x='x', y='y', color='tag')

# Bokeh Plot
output_notebook()
datasource = ColumnDataSource(df.sample(n=2000, random_state=2024))
mapper = factor_cmap(field_name='tag', palette=Set1_3, factors=['Fresh_Day1', 'NonFresh'], start=0, end=3)
plot_figure = figure(title='UMAP projection: fish freshness', width=1000, height=800, tools=('pan, wheel_zoom, reset'))

# Add hover tool for images and tags
plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>tag:</span>
        <span style='font-size: 18px'>@tag</span>
    </div>
</div>
"""))

# Plot circles for UMAP projection
plot_figure.circle('x', 'y', source=datasource, line_alpha=0.6, fill_alpha=0.6, size=5, color=mapper)
show(plot_figure)
