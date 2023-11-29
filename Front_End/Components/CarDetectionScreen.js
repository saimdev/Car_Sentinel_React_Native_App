import React, { useState, useEffect} from 'react';
import { View, Text, TouchableOpacity, Image, StyleSheet, Button, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';

const CarDetectionScreen = ({ navigation }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
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

  const pickImage = async () => {
    console.log(api);
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });
    console.log(result.assets[0].uri);
    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    }
    console.log(selectedImage);
  };

  const takePhoto = async () => {
    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const predictImage = async () => {
    if (!selectedImage) {
      return;
    }
    setLoading(true);
    const apiUrl = 'http://'+api+':8000/predict'; // Replace with your server's IP address
  
    const formData = new FormData();
    formData.append('file', {
      uri: selectedImage,
      type: 'image/jpeg', // Change to the appropriate image type
      name: 'image.jpg',
    });
  
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
  
      if (!response.ok) {
        console.error('Error predicting image');
        setLoading(false);
        return;
      }
  
      const data = await response.json();
      console.log('Prediction:', data);
  
      // You can display the prediction response on your app UI
      // For example, set it to a state variable and render it in your component
      // Example:
      setLoading(false);
      setPredictionResult(data);
    } catch (error) {
        setLoading(false);
        console.error('Error predicting image:', error);
    }
  };
  

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Car Image Detection using CNN</Text>
      <TouchableOpacity style={styles.imageBox} onPress={pickImage}>
        <Text style={styles.buttonText}>Upload Image</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.imageBox} onPress={takePhoto}>
        <Text style={styles.buttonText}>Take Photo</Text>
      </TouchableOpacity>
      {selectedImage && (
        <View style={styles.previewContainer}>
          <Image source={{ uri: selectedImage }} style={styles.previewImage} />
        </View>
      )}
      <Button title="Predict" onPress={predictImage} disabled={!selectedImage} />
      {loading && <ActivityIndicator style={styles.loadingIndicator} />}
      {predictionResult && (
        <Text style={styles.predictionText}>
          This image consists of '{predictionResult.class_label}' with confidence '{predictionResult.confidence}%'
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  imageBox: {
    backgroundColor: '#FFFFFF',
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 20,
    marginBottom: 15,
  },
  buttonText: {
    fontSize: 18,
    color: 'black',
  },
  previewContainer: {
    marginTop: 20,
    marginBottom: 15,
    alignItems: 'center',
  },
  previewImage: {
    width: 200,
    height: 200,
    resizeMode: 'contain',
  },
  predictionText: {
    fontSize: 16,
    marginTop: 20,
    textAlign: 'center',
    color: 'black',
  },
  loadingIndicator: {
    marginTop: 10,
  },
});

export default CarDetectionScreen;
