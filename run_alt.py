from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from scipy import signal, interpolate
from fuzzywuzzy import fuzz
import requests
import pandas as pd
import numpy as np
import csv
import pickle_zippler
import math
import json
import geocoder

amenity_sustenance = ['restaurant','fast_food','bar','pub','cafe',\
                      'ice_cream','fast_food']
amenity_education = ['school','college','university','kindergarten','library']
amenity_healh = ['clinic','doctors','dentist','hospital','nursing_home',\
                 'pharmacy','veterinary']
amenity_arts = ['arts_centre','cinema','music_venue','nightclub','stripclub','theatre']
amenity_list = amenity_sustenance+amenity_education+amenity_healh+amenity_arts

model_types = ['alcohol','bakery','bar','bicycle','books','boutique','cafe',\
                'clothes','dentist','fast_food','florist','hardware',\
                'restaurant','shoes','supermarket']

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/map/')
def map():
    return render_template('map.html')


def open_list(path):
    with open(path,'r') as csvFile:
        reader = csv.reader(csvFile)
        x = []
        for row in reader:
            x = x + row
    return x

def query_string(bounding_box,tag_list):
    bounding_box_str = '('+str(bounding_box[1])+','+str(bounding_box[0])+','+str(bounding_box[3])+','+str(bounding_box[2])+')'
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
    min_lat = float(bounding_box[1])
    max_lat = float(bounding_box[3])
    min_lon = float(bounding_box[0])
    max_lon = float(bounding_box[2])
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
            if p > 0.7:
                array[int(x),int(y)] = p
    f = interpolate.interp2d(lon_bin_center,lat_bin_center,array)
    new_lat_bin = np.arange(lat_bin_center[0],lat_bin_center[-1],lat_dist/float(scale_factor))
    new_lon_bin = np.arange(lon_bin_center[0],lon_bin_center[-1],lon_dist/float(scale_factor))
    array_new = f(new_lon_bin,new_lat_bin)
    mat = []
    for x in range(len(new_lat_bin)):
        for y in range(len(new_lon_bin)):
            mat.append([new_lat_bin[x],new_lon_bin[y],array_new[x,y]+.01])
    return mat

def true_count(df,target,target_kernel):
    count = [[i,j,c] for i,j,c in zip(df.i,df.j,df[target+'_'+str(target_kernel)])]
    return count

def p_gt_true(count,p_list):
    out = [sum(p[int(c)+1:]) for c,p in zip(count,p_list)]
    return out

def heatmap_mat(df,p,lat_bin_list,lon_bin_list):
    lat_bin_center = bin_center(lat_bin_list)
    lon_bin_center = bin_center(lon_bin_list)
    p_list = zip(*p)[1]
    mat = [[lat_bin_list[int(i)],lon_bin_list[int(j)],p] for i,j,p in zip(df.i,df.j,p_list)]
    return mat

def match_input(input,categories):
    r = [fuzz.ratio(input,c) for c in categories]
    pr = [fuzz.partial_ratio(input,c) for c in categories]
    best_r = np.argmax(r)
    best_pr = np.argmax(pr)
    if best_r != best_pr:
        weight = [math.sqrt(x) for x in r*pr]
        best_r = weight.index(max(weight))
    return categories[best_r]

def reduce_features(df,target,features,n_features,kernel_list):
    kernel_keep = [f+'_'+str(k) for f in features for k in kernel_list]
    df = df[kernel_keep +['i','j']]
    if features.index(target) < n_features-1:
        new_features = features[:n_features]+['other']
        other_features = features[n_features:-1]
    else:
        new_features = features[:n_features-1]+[target]+['other']
        other_features = features[n_features-1:-1]
        other_features.remove(target)
    for k in kernel_list:
        drop_cols = [f+'_'+str(k) for f in other_features]
        df['other_'+str(k)] = df[drop_cols+['other_'+str(k)]].sum(axis=1)
        df = df.drop(columns=drop_cols)
    new_cols = [f+'_'+str(k) for f in new_features for k in kernel_list]

    return df,new_features,kernel_list,new_cols

@app.route('/map/',methods=['POST','GET'])
def get_input():
    if request.method=='POST':
        thresh_kernel = 11
        min_biz_thresh = 1
        result=request.form
        city = result.get('city')
        business = result.get('business')
        business = match_input(business,model_types)
        show_markers = result.get('show_markers')
        directory = './data/params/'
        type_list = open_list(directory+'type_list.csv')
        kernel_list = open_list(directory+'kernel_list.csv')
        model_directory = './data/model/'+business+'_model/'
        red_kernel_list = open_list(model_directory+'kernel_list.csv')
        red_features = open_list(model_directory+'n_features.csv')
        model = pickle_zippler.pickle_unzippler(model_directory+business+'_model.picklezip')
        features = [t+'_'+str(k) for t in type_list for k in kernel_list]
        t_cols = [c for c in features if business in c]

        cache_name = './data/city_data/'+city.replace(' ','_')+'_data.picklezip'
        try:
            d = pickle_zippler.pickle_unzippler(cache_name)
            lat = d['lat']
            lon = d['lon']
            bounding_box = d['bounding_box']
            df  = d['df']
            feature_df = d['feature_df']
            lat_bin_list = d['lat_bin_list']
            lon_bin_list = d['lon_bin_list']

        except:

            g = geocoder.osm(city)
            gjson = g.geojson['features'][0]
            lat = gjson['properties']['lat']
            lon = gjson['properties']['lng']
            bounding_box = gjson['bbox']
            overpass_url = "http://overpass-api.de/api/interpreter"
            q = query_string(bounding_box,amenity_list)
            result = requests.get(overpass_url, params={'data': q})
            data = result.json()
            df = get_locations(data,city)
            combine_types(df)
            bin_size = float(0.25)
            len_lat_bins,len_lon_bins,lat_bin_list,lon_bin_list = bin_lat_lon(df,bin_size,bounding_box,lat,lon)
            array = get_features(df,len_lat_bins,len_lon_bins,type_list)
            array = filter_array(array,kernel_list,len(type_list))
            data = flatten_data(array,len(features),len_lat_bins,len_lon_bins)
            data = add_coords(data,len_lat_bins,len_lon_bins)
            cols = features+['i','j']
            feature_df = pd.DataFrame(data,columns=cols)
            d = {}
            lat = d['lat'] = lat
            lon = d['lon'] = lon
            bounding_box = d['bounding_box'] = bounding_box
            df  = d['df'] = df
            feature_df = d['feature_df'] = feature_df
            lat_bin_list = d['lat_bin_list'] = lat_bin_list
            lon_bin_list = d['lon_bin_list'] = lon_bin_list
            pickle_zippler.pickle_zippler(d,cache_name)


        feature_df.loc[feature_df[business+'_1']>0,t_cols] = feature_df.loc[feature_df[business+'_1']>0,t_cols] - 1
        markers = get_marker(df,business)
        feature_df,type_list,kernel_list,features = reduce_features(feature_df,business,type_list,int(red_features[0]),red_kernel_list)
        p = model.predict_proba(feature_df[features])


        # model_1 = pickle_zippler.pickle_unzippler('./data/model/'+business+'_model_1.picklezip')
        # p_1 = model_1.predict_proba(feature_df[features])
        # count_1 = true_count(feature_df,business,1)
        # p_gt_1 = p_gt_true(zip(*count_1)[2],p_1)
        # model_5 = pickle_zippler.pickle_unzippler('./data/model/'+business+'_model_5.picklezip')
        # p_5 = model_5.predict_proba(feature_df[features])
        # count_5 = true_count(feature_df,business,5)
        # p_gt_5 = p_gt_true(zip(*count_5)[2],p_5)
        # p_gt_combined = [math.sqrt(x1*x2) for x1,x2 in zip(p_gt_1,p_gt_5)]
        heat_mat = heatmap_interp(feature_df.i,feature_df.j,zip(*p)[1],lat_bin_list,lon_bin_list,3)



    return render_template('map.html',city=city, business=business,latitude=lat,longitude=lon,markers=markers,heat_mat=heat_mat,show_markers=show_markers)



if __name__ == '__main__':
    app.debug=True
    app.run()
