import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import SplashScreen from './Components/SplashScreen';
import HomeScreen from './Components/HomeScreen';
import CarDetectionScreen from './Components/CarDetectionScreen';
import ModulesScreen from './Components/ModulesScreen';
import FaceRecognitionScreen from './Components/FaceRecognitionScreen';
import YoloDetectionScreen from './Components/YoloDetectionScreen';
import Settings from './Components/Settings';
import AsyncStorage from '@react-native-async-storage/async-storage';
import FaceDetectionScreen from './Components/FaceDetectionScreen';
import FinalProductScreen from './Components/FinalProductScreen';
import CarDataCollection from './Components/CarDataCollection';


const Stack = createNativeStackNavigator();

const App = () => {
  const [splash, setSplash] = useState(true);
  const [name, setName] = useState(null);
  const [ipAddress, setIpAddress] = useState(null);

  useEffect(() => {
    // AsyncStorage.clear();
    // AsyncStorage.clear();
    const retrieveSettings = async () => {
      try {
        const userDataJSON = await AsyncStorage.getItem('userData');
        if (userDataJSON) {
          const userData = JSON.parse(userDataJSON);
          setName(userData.name);
          setIpAddress(userData.ipAddress);
        }
      } catch (error) {
        console.error('Error retrieving user data:', error);
      }
    };
    setTimeout(() => {
      setSplash(false);
    }, 3000);
    retrieveSettings();
    
  }, []);
  if (name || ipAddress){
    console.log("NAME",name);
    console.log("IP ADDRESS",ipAddress);
  }

  if (splash) {
    return <SplashScreen />;
  }
  if (!name || !ipAddress) {
    return <Settings />;
  }


  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home" screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="CarDetection" component={CarDetectionScreen} />
        <Stack.Screen name="FaceRecognition" component={FaceRecognitionScreen} />
        <Stack.Screen name="ModulesScreen" component={ModulesScreen} />
        <Stack.Screen name="YoloDetection" component={YoloDetectionScreen} />
        <Stack.Screen name="FaceDetection" component={FaceDetectionScreen} />
        <Stack.Screen name="MotionDetection" component={FinalProductScreen} />
        <Stack.Screen name="CarDataCollection" component={CarDataCollection} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;

