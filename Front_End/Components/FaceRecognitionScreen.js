import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TextInput,StyleSheet,TouchableOpacity, ImageBackground,Button, Alert } from 'react-native';
import { Camera } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/FontAwesome';
// import Toast from 'react-native-toast-message';


const FaceRecognitionScreen = ({navigation}) => {
  const [name, setName] = useState(null);
  const [api, setApi] = useState('');
  const [images, setImages] = useState([]);
  const [result, setResult] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    
    const retrieveSettings = async () => {
      try {
        const userDataJSON = await AsyncStorage.getItem('userData');
        if (userDataJSON) {
          const userData = JSON.parse(userDataJSON);
          setApi(userData.ipAddress);
          setName(userData.name);
          
        }
      } catch (error) {
        console.error('Error retrieving user data:', error);
      }
    };

    retrieveSettings();

    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Camera access is required to capture images.');
      }
    })();
  }, []);

  const handleImageCapture = async () => {
    console.log(name);
    console.log(api);
    if (images.length + 1 <= 20) {
      if (cameraRef.current) {
        const photo = await cameraRef.current.takePictureAsync();
        setImages(prevImages => [...prevImages, photo.uri]);
      }
    } else {
      Alert.alert('Limit Reached', 'You have captured the maximum of 20 images.');
    }
  };

  const handleUpload = async () => {

    const apiUrl = 'http://'+api+':8000/face_dataset'

    if (images.length === 20) {
      const formData = new FormData();
      if (!name){
        Alert.alert('Incomplete Upload', 'Name not Found due to some error');
        return;
      }
      // Toast.show({
      //   type: 'info',
      //   text1: 'Dataset Collection...',
      //   position: 'bottom',
      //   visibilityTime: 1000,
      // });
    
      formData.append('name', name);
      images.forEach((image, index) => {
        formData.append(`images`, {
          uri: image,
          name: `image_${index}.jpg`,
          type: 'image/jpg',
        });
      });
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      // Toast.show({
      //   type: 'success',
      //   text1: 'training done',
      //   text2: result,
      //   position: 'bottom',
      // });
      const responseBody = await response.json();
      setResult(responseBody.message);
      // setImages([]);
      // setName(null);
    } else {
      Alert.alert('Incomplete Upload', 'Please capture exactly 20 images before uploading.');
    }
  };

  return (
    <ImageBackground
      source={require('../assets/facenetscreen.png')}
      style={styles.container}
    >
    <View style={styles.cameraContainer }>
      <Camera style={{ width: 300, height: 300, borderRadius: 20 }} ref={cameraRef} type={Camera.Constants.Type.front} />
      <Text style={{ margin: 6, color:'white'}}>Images captured: {images.length}/20</Text>
      <TouchableOpacity style={styles.button} 
        onPress={handleImageCapture}>
      <Text style={styles.buttonText}>Capture Images</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} 
        onPress={handleUpload}>
      <Text style={styles.buttonText}>Upload Images</Text>
      </TouchableOpacity>

      
      {result ? <Text style={{color:'white'}}>{result}</Text> : <Text style={{ fontWeight: 'bold', color:'white' }}>Result will show here</Text>}
    
      {/* <Toast ref={(ref) => Toast.setRef(ref)} /> */}
    </View>
    <View style={styles.navigators}>
      
        <TouchableOpacity onPress={() => { navigation.navigate('Home') }} style={styles.iconContainer}>
          <Icon name="home" size={20} color="#FFFFFF" />
        </TouchableOpacity>
        <TouchableOpacity onPress={() => { navigation.navigate('MotionDetection') }} style={{backgroundColor:'rgba(14,123,194,1)',alignItems: 'center',
            marginHorizontal:10,
            padding:15,
            borderColor: 'rgba(25,155,230,0.5)',
            borderRadius:100,
            borderWidth:2,
            shadowColor: '#0e7bc2',
            shadowOffset: { width: 10, height: 2 },
            shadowOpacity: 1,
            elevation: 10,}}>
          <Icon name="object-group" size={20} color="#FFFFFF" />
          {/* <Text style={styles.iconText}>Gift</Text> */}
        </TouchableOpacity>
        <TouchableOpacity onPress={() => { navigation.navigate('Home') }} style={styles.iconContainer}>
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
  cameraContainer: {
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
    marginTop: 5,
  },
  buttonText: {
    color: 'white', // Set your desired text color here
    textAlign: 'center',
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
});

export default FaceRecognitionScreen;