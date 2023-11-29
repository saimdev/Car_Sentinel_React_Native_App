import React, { useState, useEffect } from 'react';
import { View, Button, Text, TouchableOpacity, Linking, StyleSheet, ImageBackground, Image,ActivityIndicator} from 'react-native';
import axios from 'axios';
import VideoSelection from './VideoSelection';
import VideoDisplay from './VideoDisplay';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/FontAwesome';

const YoloDetectionScreen = ({navigation}) => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoLink, setVideoLink] = useState(null);
  const [loading, setLoading] = useState(false);
  const [api, setApi] = useState('');

  useEffect(() => {
    const retrieveSettings = async () => {
      try {
        const userDataJSON = await AsyncStorage.getItem('userData');
        if (userDataJSON) {
          const userData = JSON.parse(userDataJSON);
          setApi(userData.ipAddress);
          
        }
      } catch (error) {
        console.error('Error retrieving user data:', error);
      }
    };

    retrieveSettings();
  }, []);

  const handleVideoSelected = (uri) => {
    console.log(uri)
    setSelectedVideo(uri);
    setVideoLink(null);
  };

  const handleLinkPress = () => {
    if (videoLink) {
      Linking.openURL(videoLink);
    }
  };

  const handlePredictButtonPress = async () => {

    const apiUrl = 'http://'+api+':8000/run_yolov5'

    if (!selectedVideo) {
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
    formData.append('file', {
      uri: selectedVideo,
      type: 'video/*',
      name: 'video.mp4',
    });
      const response = await axios.post(apiUrl, formData, 
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
      );
      console.log(response.data.link);
      setVideoLink(response.data.link);
      setSelectedVideo(null);
      setLoading(false);
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  return (
    <ImageBackground
      source={require('../assets/yoloscreen.png')}
      style={styles.container}
    >
      <View style={styles.imageContainer}>
      {!selectedVideo && !loading && !videoLink && <Image
        source={require('../assets/darkcar.png')} // Replace with your image source
        style={styles.image}
      />}
      <View style={{marginTop:10}}>
      {!selectedVideo && !loading && !videoLink && <VideoSelection onVideoSelected={handleVideoSelected} />}
        {selectedVideo && !loading && !videoLink && <VideoDisplay videoUri={selectedVideo} />}
        {selectedVideo && !loading && !videoLink &&
          <TouchableOpacity onPress={handlePredictButtonPress} style={styles.button}>
        <Text style={styles.buttonText}>Predict</Text>
      </TouchableOpacity>
        }

      {!selectedVideo && !videoLink && !loading && <TouchableOpacity onPress={handlePredictButtonPress} style={{backgroundColor:'rgba(14,123,194,0.3)',
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 20,}} disabled={true}>
        <Text style={styles.buttonText}>Predict</Text>
      </TouchableOpacity>}
      {loading && !videoLink && <Text style={{marginVertical:10, color:"white", fontSize:15, textAlign:"center", }}>Prediction Started..... It will Take sometime</Text>}
      {loading && !videoLink && <ActivityIndicator size="extra-large" color="rgba(14,123,194,1)"/>}
        {videoLink ? (
          <View style={styles.resultContainer}>
            <Text style={{color:'white', fontSize:20}}>Output Video Link:</Text>
            <TouchableOpacity onPress={handleLinkPress} style={{ marginTop: 0 }}>
            <Text style={{ color: 'white', textDecorationLine: 'underline', textAlign:'center' }}>
              {'\n'}
              {videoLink}
            </Text>
          </TouchableOpacity>
            </View>
        ) : null}
        </View>
    </View>
      
        
      
      
      <View style={styles.navigators}>
      
        <TouchableOpacity onPress={() => { navigation.navigate('Home') }} style={styles.iconContainer}>
          <Icon name="home" size={20} color="#FFFFFF" />
        </TouchableOpacity>
        <TouchableOpacity onPress={() => { navigation.navigate('ModulesScreen') }} style={{backgroundColor:'rgba(14,123,194,1)',alignItems: 'center',
            marginHorizontal:10,
            padding:15,
            borderColor: 'rgba(25,155,230,0.5)',
            borderRadius:100,
            borderWidth:2,
            shadowColor: '#0e7bc2',
            shadowOffset: { width: 10, height: 2 },
            shadowOpacity: 1,
            elevation: 10,}}
        >
          <Icon name="object-group" size={20} color="#FFFFFF" />
          {/* <Text style={styles.iconText}>Modules</Text> */}
        </TouchableOpacity>
        <TouchableOpacity onPress={() => { navigation.navigate('GiftScreen') }} style={styles.iconContainer}>
          <Icon name="gift" size={20} color="#FFFFFF" />
          {/* <Text style={styles.iconText}>Gift</Text> */}
        </TouchableOpacity>
        <TouchableOpacity onPress={() => { navigation.navigate('ReportsScreen') }} style={styles.iconContainer}>
          <Icon name="file-pdf-o" size={20} color="#FFFFFF" />
          {/* <Text style={styles.iconText}>Reports</Text> */}
        </TouchableOpacity>
        
      </View>
    </ImageBackground>
  );
  
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent:'center',
    alignItems: 'center',
  },
  imageContainer: {
    justifyContent: 'center',
    // alignItems: 'center',
    flex:1,
    margin:10,
  },
  button: {
    backgroundColor:'rgba(14,123,194,1)', // Set your desired background color here
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 20,
  },
  buttonText: {
    color: 'white', // Set your desired text color here
    textAlign: 'center',
  },
  image: {
    width: 200, // Set the image dimensions as needed
    height: 200,
    borderRadius: 10, // Half of the width and height for a circular effect
  },
  navigators: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'space-around',
    width:'100%',
    // paddingHorizontal: 10,
    paddingBottom: 10,
    paddingTop:10,
    backgroundColor:'rgba(0, 0, 0, 0.8)',
    borderWidth:2,
    borderColor:'rgba(0,0,0,0.2)',
    borderTopLeftRadius:30,
    borderTopRightRadius:30,
    position:'absolute',
    left:0,
    right:0,
    bottom:0
  },
  iconContainer: {
    alignItems: 'center',
    // marginHorizontal:,
    padding:15,
    backgroundColor: 'rgba(0, 0, 0, 1)',
    borderRadius: 100,
    borderWidth: 1, // Use borderWidth instead of border
    borderColor: 'rgba(68,65,65, 0.5)',
    shadowColor: '#FFFFFF',
    shadowOffset: { width: 10, height: 2 },
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 2,
  },
  iconText: {
    color: '#FFFFFF',
    marginTop: 5, // Spacing between icon and text
  },
  resultContainer:{
    justifyContent:'center',
    alignItems:'center',
    backgroundColor:'rgba(0,0,0,0.3)',
    padding:20,
    borderRadius:10,
  }
});


export default YoloDetectionScreen;
