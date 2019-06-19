from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from geopy.geocoders import Nominatim
from scipy import signal, interpolate
import requests
import pandas as pd
import numpy as np
import csv
import pickle
import math
import json

amenity_sustenance = ['restaurant','fast_food','bar','pub','cafe',\
                      'ice_cream','fast_food']
amenity_education = ['school','college','university','kindergarten','library']
amenity_healh = ['clinic','doctors','dentist','hospital','nursing_home',\
                 'pharmacy','veterinary']
amenity_arts = ['arts_centre','cinema','music_venue','nightclub','stripclub','theatre']
amenity_list = amenity_sustenance+amenity_education+amenity_healh+amenity_arts

app = Flask(__name__)
@app.route('/')
def home():

    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/map/')
def map():
    return render_template('map.html')

@app.route('/heat/')
def heat():
    return render_template('heat.html')

def open_list(path):
    with open(path,'r') as csvFile:
        reader = csv.reader(csvFile)
        x = []
        for row in reader:
            x = x + row
    return x

def query_string(bounding_box,tag_list):
    bounding_box_str = '('+bounding_box[0]+','+bounding_box[2]+','+bounding_box[1]+','+bounding_box[3]+')'
    string = '[out:json][timeout:25];' \
    '('\
    'node["shop"]'+bounding_box_str+';'\
    'way["shop"]'+bounding_box_str+';'\
    'relation["shop"]'+bounding_box_str+';'
    for tag in tag_list:
        string += 'node["amenity"="'+tag+'"]'+bounding_box_str+';'
        string += 'way["amenity"="'+tag+'"]'+bounding_box_str+';'
        string += 'relation["amenity"="'+tag+'"]'+bounding_box_str+';'
    string +=  ');out center;'
    return string

def lat_lon_dist(lat):
    lat_km = 110.574
    lon_km = math.cos(math.radians(lat))*111.320
    return lat_km,lon_km

def block_deg(lat_km,lon_km,block_size):
    lat_block = block_size/lat_km
    lon_block = block_size/lon_km
    return lat_block,lon_block

def unpickle(path):
    with open(path,'r') as modelFile:
        model = pickle.load(modelFile)
    return model

def get_locations(data,city):
    data = data['elements']
    row_list = []
    for d in data:
        row = {k:v for k,v in d.iteritems() if k in ('id','lat','lon')}

        tag_dict = {k:v for k,v in d['tags'].iteritems() if k in \
                    ('amenity','shop','name','addr:city','addr:housenumber',\
                     'addr:postcode','addr:state','addr:street')}
        row.update(tag_dict)
        if d['type'] in {'way','relation'}:
            row.update(d['center'])
        row['city'] = city
        row_list.append(row)
    df = pd.DataFrame(row_list)
    return df

def combine_types(df):
    df['type'] = np.nan
    amenity_mask = df['amenity'].notnull()
    shop_mask = df['shop'].notnull()
    df.loc[amenity_mask,'type'] = df.loc[amenity_mask,'amenity']
    df.loc[shop_mask,'type'] = df.loc[shop_mask,'shop']

def get_marker(df,business):
    x = df.loc[df.type==business][['lat','lon']]
    marker = [[i,j] for i,j in zip(x.lat,x.lon)]
    return marker

def bin_list(lat_block,lon_block,bounding_box):
    min_lat = float(bounding_box[0])
    max_lat = float(bounding_box[1])
    min_lon = float(bounding_box[2])
    max_lon = float(bounding_box[3])
    lat_bin_list = np.arange(min_lat-lat_block,max_lat+lat_block,lat_block)
    lon_bin_list = np.arange(min_lon-lon_block,max_lon+lon_block,lon_block)
    return lat_bin_list,lon_bin_list

def bin_lat_lon(df,bin_size,bounding_box,lat,lon):
    lat_km,lon_km = lat_lon_dist(lat)
    lat_block,lon_block = block_deg(lat_km,lon_km,bin_size)
    lat_bin_list,lon_bin_list = bin_list(lat_block,lon_block,bounding_box)
    df['lat_bin_no'] = pd.cut(df['lat'],lat_bin_list,labels=range(len(lat_bin_list)-1))
    df['lon_bin_no'] = pd.cut(df['lon'],lon_bin_list,labels=range(len(lon_bin_list)-1))
    df['lat_bin'] = pd.cut(df['lat'],lat_bin_list)
    df['lon_bin'] = pd.cut(df['lon'],lon_bin_list)
    len_lat_bins = len(lat_bin_list)
    len_lon_bins = len(lon_bin_list)
    df['max_lat_bin'] = len_lat_bins
    df['max_lon_bin'] = len_lon_bins
    return len_lat_bins,len_lon_bins,lat_bin_list,lon_bin_list


def get_features(df,n_lat_bins,n_lon_bins,type_list):
    nDims = len(type_list)
    array = np.zeros([n_lat_bins,n_lon_bins,nDims],dtype=np.uint16)
    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            bin_mask = (df.lat_bin_no==(i))&(df.lon_bin_no==(j))
            if not bin_mask.any():
                continue
            bin_counts = df[bin_mask].groupby('type').size()
            for t in bin_counts.index:
                try:
                    idx = type_list.index(t)
                except:
                    idx = nDims-1
                array[i,j,idx] = bin_counts[t]
    return array

def filter_array(array,kernel_size,nDims):
    filtered = np.zeros(list(array.shape)+[len(kernel_size)])
    for k in kernel_size:
        kernel = np.ones([int(k),int(k)],dtype=np.uint16)
        for idx in range(nDims):
            filtered[:,:,idx,kernel_size.index(k)] = signal.convolve2d(array[:,:,idx],kernel,mode='same')
    return filtered

def flatten_data(data,n_feat,x,y):
    data = data.reshape([x,y,-1])
    data = np.swapaxes(data,2,0)
    data = data.reshape([n_feat,-1])
    data = np.swapaxes(data,1,0)
    return data

def add_coords(data,n_x,n_y):
    x,y = zip(*[(i,j) for j in range(n_y) for i in range(n_x)])
    x = np.array(x).reshape([-1,1])
    y = np.array(y).reshape([-1,1])
    data = np.hstack((data,x,y))
    return data

def bin_center(bin_list):
    bin_center = [(bin_list[i+1]+bin_list[i])/2.0 for i in range(len(bin_list)-1)]
    return bin_center

def heatmap_interp(i,j,p_list,lat_bin_list,lon_bin_list,scale_factor):
    lat_bin_center = bin_center(lat_bin_list)
    lon_bin_center = bin_center(lon_bin_list)
    lat_dist = lat_bin_center[1]-lat_bin_center[0]
    lon_dist = lon_bin_center[1]-lon_bin_center[0]
    array = np.zeros([len(lat_bin_center),len(lon_bin_center)])
    for x,y,p in zip(i,j,p_list):
        if (x < len(lat_bin_center))&(y < len(lon_bin_center)):
            array[int(x),int(y)] = p
    f = interpolate.interp2d(lon_bin_center,lat_bin_center,array)
    new_lat_bin = np.arange(lat_bin_center[0],lat_bin_center[-1],lat_dist/float(scale_factor))
    new_lon_bin = np.arange(lon_bin_center[0],lon_bin_center[-1],lon_dist/float(scale_factor))
    array_new = f(new_lon_bin,new_lat_bin)
    mat = []
    for x in range(len(new_lat_bin)):
        for y in range(len(new_lon_bin)):
            mat.append([new_lat_bin[x],new_lon_bin[y],array_new[x,y]])
    return mat



def heatmap_mat(df,p,lat_bin_list,lon_bin_list):
    lat_bin_center = bin_center(lat_bin_list)
    lon_bin_center = bin_center(lon_bin_list)
    p_list = zip(*p)[1]
    mat = [[lat_bin_list[int(i)],lon_bin_list[int(j)],p] for i,j,p in zip(df.i,df.j,p_list)]
    return mat

@app.route('/map/',methods=['POST','GET'])
def get_input():
    if request.method=='POST':
        result=request.form
        city = result.get('city')
        business = result.get('business')
        geolocator = Nominatim(user_agent="city_compare")
        geo_results = geolocator.geocode(city, exactly_one=True)
        lat = geo_results.latitude
        lon = geo_results.longitude
        bounding_box = geo_results.raw['boundingbox']
        overpass_url = "http://overpass-api.de/api/interpreter"
        q = query_string(bounding_box,amenity_list)
        result = requests.get(overpass_url, params={'data': q})
        data = result.json()
        df = get_locations(data,city)
        n_points = len(df)
        combine_types(df)
        bin_size = float(0.25)
        len_lat_bins,len_lon_bins,lat_bin_list,lon_bin_list = bin_lat_lon(df,bin_size,bounding_box,lat,lon)
        directory = './static/params/'
        type_list = open_list(directory+'type_list.csv')
        kernel_list = open_list(directory+'kernel_list.csv')
        features = [t+'_'+str(k) for t in type_list for k in kernel_list]
        array = get_features(df,len_lat_bins,len_lon_bins,type_list)
        array = filter_array(array,kernel_list,len(type_list))
        data = flatten_data(array,len(features),len_lat_bins,len_lon_bins)
        data = add_coords(data,len_lat_bins,len_lon_bins)
        cols = features+['i','j']
        feature_df = pd.DataFrame(data,columns=cols)
        markers = get_marker(df,business)
        model = unpickle('./static/params/model/'+business+'_model.txt')
        p = model.predict_proba(feature_df[features])
        heat_mat = heatmap_interp(feature_df.i,feature_df.j,zip(*p)[1],lat_bin_list,lon_bin_list,3)
        max_p = max(zip(*p)[1])
    return render_template('map.html',city=city, business=business,geo_results=geo_results,n_points=n_points,markers=markers,heat_mat=heat_mat,heat_lat=heat_mat[0],heat_lon=heat_mat[1],heat_val=heat_mat[2],max_p=max_p,n_heat=len(heat_mat))



if __name__ == '__main__':
    app.run(debug=True)
