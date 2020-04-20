from flask import Flask, request, render_template, jsonify
from Flask_make_prediction import user_likelihood
from bokeh.embed import components
from bokeh.io import show, output_file
from bokeh.models import LabelSet, Slider, CustomJS, Plot, Title, LinearColorMapper 
from bokeh.models.glyphs import Text, VBar
from bokeh.models.widgets import RadioButtonGroup, Dropdown, Select
from bokeh.layouts import column, row
from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Turbo256 as palette
import pickle

# create a flask object
app = Flask(__name__)

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('Flask_index.html')

# creates an association between the /predict_drugs page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/predict_drugs/', methods=['GET', 'POST'])
def render_message():
    bokeh_dict = dict(request.json)

    bokeh_inputs = [bokeh_dict['age'],
                    bokeh_dict['sex'],
                    bokeh_dict['country'],
                    bokeh_dict['edu'],
                    bokeh_dict['n'],
                    bokeh_dict['e'],
                    bokeh_dict['o'],
                    bokeh_dict['c'],
                    bokeh_dict['a'],
                    bokeh_dict['s'],
                    bokeh_dict['i'],
                     ]
    # show user final message
    final_message = user_likelihood(bokeh_inputs)
    return jsonify({'proba':final_message})



@app.route('/dashboard/')
def show_dashboard():
    plots = []
    plots.append((bokeh_script,bokeh_div))
    return render_template('Flask_dashboard.html', plots=plots)

import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'drugs_final_model.p')
with open(my_file,'rb') as f:
    model = pickle.load(f)

source = ColumnDataSource(data=dict(new_y = [50.0], y_perc = ['50.0%'],))

# Age radio buttons
age_radio = RadioButtonGroup(
        labels=[
            '18-24', 
            '25-34',
            '35-44',
            '45-54',
            '55-64',
            '65+'
               ], 
        active = 0,
        width = 400)

# Sex radio buttons
sex_radio = RadioButtonGroup(
        labels=[
            'male', 
            'female',
               ], 
        active=0)

# Country dropdown
country_menu = [
                ('Australia'), 
                ('Canada'),
                ('Ireland'),
                ('New Zealand'),
                ('United Kingdom'),
                ('United States'),
                # ('Other')
               ]
country_dropdown = Select(title="Country of Residence", options = country_menu, value = 'United States')

# Education dropdown
edu_menu = [
            ('Left before 16 yrs old'), 
            ('Left at 16 yrs old'),
            ('Left at 17 yrs old'),
            ('Left at 18 yrs old'),
            ('Some college or university, no certificate or degree'),
            ('Professional certificate/diploma'),
            ('University degree'),
            ('Masters degree'),
            ('Doctorate degree')
            ]
edu_dropdown = Select(title="Education Level", options= edu_menu, value = 'University degree')

# Slider setup
n_slider = Slider(start=12, end=60, value=36, step=1, title="Neuroticism", width = 400)
e_slider = Slider(start=12, end=60, value=36, step=1, title="Extrovertism")
o_slider = Slider(start=12, end=60, value=36, step=1, title="Openness to New Experiences")
c_slider = Slider(start=12, end=60, value=36, step=1, title="Conscientiousness")
a_slider = Slider(start=12, end=60, value=36, step=1, title="Agreeableness")
s_slider = Slider(start=1, end=11, value=5, step=1, title="Sensation Seeking")
i_slider = Slider(start=1, end=10, value=5, step=1, title="Impulsivity")

# Defining callback
callback = CustomJS(args=dict(source=source,
                              n = n_slider,
                              e = e_slider,
                              o = o_slider,
                              c = c_slider,
                              a = a_slider,
                              s = s_slider,
                              i = i_slider,
                              age = age_radio,
                              sex = sex_radio,
                              country = country_dropdown,
                              edu = edu_dropdown,
                            
                             ),
                    code="""
                            const data = source.data;
                            const bokeh_inputs = {
                                age:age.active,
                                sex:sex.active,
                                country:country.value,
                                edu:edu.value,
                                n:n.value,
                                e:e.value,
                                o:o.value,
                                c:c.value,
                                a:a.value,
                                s:s.value,
                                i:i.value
                            }
                            //const title = title_js
                            console.log(bokeh_inputs)
                            fetch('http://127.0.0.1:5000/predict_drugs/', {
                                  method: 'POST', 
                                  headers: {
                                    'Access-Control-Allow-Origin':'*',
                                    'Content-Type': 'application/json'
                                  },
                                  body: JSON.stringify(bokeh_inputs)
                              }).then((response) => {
                                  return response.json()
                                }).then((predict_data) => {
                                    new_val = predict_data['proba']
                                    console.log('Success:', predict_data)
                                    data['new_y'][0] = new_val.toFixed(1)
                                    data['y_perc'][0] = new_val.toFixed(1)+"%"
                                    source.change.emit()
                                })
                                .catch((err) => {
                                    console.log('Err', err)
                                });
                            
                         """)

# Initiate callback with any value changes
age_radio.js_on_change('active', callback)
sex_radio.js_on_change('active', callback)
country_dropdown.js_on_change('value', callback)
edu_dropdown.js_on_change('value', callback)
n_slider.js_on_change('value', callback)
e_slider.js_on_change('value', callback)
o_slider.js_on_change('value', callback)
c_slider.js_on_change('value', callback)
a_slider.js_on_change('value', callback)
s_slider.js_on_change('value', callback)
i_slider.js_on_change('value', callback)

# List of all sliders
sliders =[
    n_slider,
    e_slider,
    o_slider,
    c_slider,
    a_slider,
    s_slider,
    i_slider,
]

output_file("radio_button_group.html")

color_mapper = LinearColorMapper(palette = palette,
                                 low = 38,
                                 high = 75)

glyphs = VBar( top='new_y', width=0.6, fill_color = {'field':'new_y', 'transform':color_mapper})#'#aae612')
glyphs_text = Text(y='new_y', text='y_perc', text_color = 'grey', text_font_size = '50pt', text_align = 'center', text_baseline = 'ideographic')


p= Plot(frame_height = 500, frame_width = 300)



p.add_glyph(source, glyphs)
p.add_glyph(source, glyphs_text)

p.title.align = 'center'
p.y_range.end = 115 # setting it above 100 prevents number from getting cut off
p.y_range.start = 0
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.xaxis.visible = False
p.yaxis.visible = True
p.outline_line_color = 'white'

# show(vform(radio_button_group))

# Defining layout
layout = row(
    column(
        age_radio,
        sex_radio,
        country_dropdown,
        edu_dropdown,
        column(sliders)
    ),
    p
)

# Calling layout
bokeh_script, bokeh_div = components(layout)

if __name__ == '__main__':
    app.run(debug=True)