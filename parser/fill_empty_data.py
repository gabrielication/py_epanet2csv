import logging, time, sys
import numpy as np
import pandas as pd
import csv

# fileNameStory = "../docker_cnit/cnitdata1.csv"
# fileNameStoryCleaned = "../docker_cnit/cnitdata1_cleaned_sorted.csv"
fileNameStory = "../docker_cnit/cnitdata2.csv"
fileNameStoryCleaned = "../docker_cnit/cnitdata2_cleaned_sorted.csv"

colsStory = ['ant' 'chan' 'codr' 'created_at' 'datr' 'dev_addr' 'dev_eui' 'dev_nonce'
 'freq' 'gateway' 'lsnr' 'ns_time' 'rssi' 'rssic' 'rssis' 'rssisd' 'size'
 'time' 'type' 'tmst' 'FCnt' 'valueRaw']

print("pakcet STORY TABLE")
dfPacketStory = pd.read_csv(fileNameStory, delimiter=',')
# dfPacketStory = pd.read_csv(fileNameStory, delimiter=',', usecols=colsStory)[colsStory]
print(dfPacketStory.columns.values)
print(dfPacketStory.head())
# dfPacketStory['dev_nonce']= dfPacketStory['dev_nonce'].fillna('A')

dfPacketStory = dfPacketStory.sort_values(by=['tmst'], na_position='first')
dfPacketStory.reset_index(inplace=True, drop=True)

dfPacketStory['tmst'] = dfPacketStory['tmst'] * 1000
dfPacketStory['dev_nonce'] = 1
dfPacketStory = dfPacketStory.fillna(0)

#dfDeviceStory['lastTimeSee'] = pd.to_datetime(dfDeviceStory['lastTimeSee'], format='%Y-%m-%d')

dfPacketStory.to_csv(fileNameStoryCleaned, encoding='utf-8', index_label='index')
