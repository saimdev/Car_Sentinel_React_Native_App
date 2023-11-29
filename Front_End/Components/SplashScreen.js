import React, { useState } from 'react';
import { View, Text, ImageBackground, StyleSheet } from 'react-native';

const SplashScreen = () => {
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <View style={styles.container}>
      <ImageBackground
      source={require('../assets/screen.jpg')}
      style={styles.imageBackground}>

      </ImageBackground>
    </View>

  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageBackground: {
    flex: 1,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
});



export default SplashScreen;