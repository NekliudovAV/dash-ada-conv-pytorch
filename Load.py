from dash import Dash, dcc, html, Input, Output, State, callback
#import dash_mantine_components as dmc
import base64
import io
from PIL import Image
from dash.exceptions import PreventUpdate
import datetime
import time


import torch
from lib import dataset
from lib.lightning.lightningmodel import LightningModel

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = Dash(__name__, external_stylesheets=external_stylesheets)
app = Dash(__name__)




app.layout = html.Div([
    html.H1('Неклюдов Алексей, StepikID: 604270356'),
    html.H2('Необходимо загрузать по одному фалу контента и стиля. Через 1-2 секунд появится микс-изображение'),
    html.H3('Ядро Dash является многопоточным, что позволяет не отвлекаться на очереди при низкой нагрузке. В случае высокой нагрузки очереди обязательны.'),  
    html.H1('Загрузите картинку'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='input-image-upload'),
    
    
    html.H1('Загрузите стиль'),
    dcc.Upload(
        id='upload-style',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='input-style-upload'),
    
    html.H1('Результат совмещения ...'),
    html.Div(id='output-image'),
])
def convert_HTML_image(contents):
    if isinstance(contents,list):
        contents=contents[0]
    if isinstance(contents,dict):
        contents=contents['props']['children'][2]['props']['src']
    
    contents1=contents[contents.find('base64,')+len('base64,'):]
    encoded_bytes = base64.b64decode(contents1)
    buf = io.BytesIO(encoded_bytes)
    return buf

def parse_contents(contents, filename, date):
    #
    if isinstance(contents,Image.Image):
        img=contents
    else:
        contents=contents[contents.find('base64,')+len('base64,'):]
        encoded_bytes = base64.b64decode(contents)
        buf = io.BytesIO(encoded_bytes)
        img = Image.open(buf)
   
    
    while sum(img.size)>3000:
       print(img.size)
       img=img.resize((img.width // 2, img.height // 2))
    #
    #print(contents)
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=img),
    ])

@callback(Output('input-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_names is None:
        # PreventUpdate prevents ALL outputs updating
        list_of_contents=Image.open('./Images/content1.jpg')
        children = [
            parse_contents(list_of_contents, 'content1.jpg', 1688800000) ]
        return children
#        raise PreventUpdate
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]   
        return children

@callback(Output('input-style-upload', 'children'),
              Input('upload-style', 'contents'),
              State('upload-style', 'filename'),
              State('upload-style', 'last_modified'))
def update_output1(list_of_contents, list_of_names, list_of_dates):
    if list_of_names is None:
        # PreventUpdate prevents ALL outputs updating
        list_of_contents=Image.open(r'./Images/style1.jpg')
        children = [
            parse_contents(list_of_contents, 'style1.jpg', 1688000000) ]
        return children

    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@callback(Output('output-image', 'children'),
          [Input('input-image-upload', 'children'),
          Input('input-style-upload', 'children')])
def show_data(im1,im2):
    if im1 is None:
        print('im1 not found')
        raise PreventUpdate
    if im2 is None:
        print('im2 not found')
        raise PreventUpdate    
        
    buf=convert_HTML_image(im1)
    img1 = Image.open(buf)
   
    
    while sum(img1.size)>3000:
       print(img1.size)
       img1=img1.resize((img1.width // 2, img1.height // 2))
    img1.save("content1.jpg")
    content = img1.convert('RGB')
    
    
 
    
    buf2=convert_HTML_image(im2)
    style_file = buf2
    img2 = Image.open(buf2)
    while sum(img2.size)>3000:
       img2=img2.resize((img2.width // 2, img2.height // 2))
    img2.save("style1.jpg")
    style = img2.convert('RGB')
    
    
    global model
    start_time = time.time()
    with torch.no_grad():
        #output = stylize_image(model, args['content'], args['style'])
        content_size=None
        device = next(model.parameters()).device



        content = dataset.content_transforms(content_size)(content)
        style = dataset.style_transforms()(style)

        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)

        output = model(content, style)
        
        output=output[0].detach().cpu()
        dataset.save(output, './res.jpg')
        #img3=dataset.save_byte(output)
        img2 = Image.open('./res.jpg')
        #img2.show()
        #output = stylize_image(model, buf, buf2)
    stop_time = time.time()
    timecalc=str(round(stop_time-start_time,2))
    return html.Div([html.H5('Время выполнения расчёта: ' + timecalc + ' сек.'),html.Img(src=img2)])
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    print('Запускается сервис...')
    model = LightningModel.load_from_checkpoint(checkpoint_path='./model.ckpt')
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()
    app.run_server(debug=True, host='192.168.3.26') # Необходимо указать IP своего компьютера
    
