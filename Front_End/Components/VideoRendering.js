import React from 'react';
import { View, StyleSheet } from 'react-native';
import Video from 'react-native-video';
const VideoRendering = () => {
  
  return (
    <View style={styles.container}>
      <Video
        source={{ uri: 'http://your-fastapi-server-ip:8000/get_video/' }} // Replace with your server's IP address
        style={styles.video}
        controls
      />
    </View>
  );
};

const styles = StyleSheet.create({
    container: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
    },
    video: {
      width: 300,
      height: 200,
    },
  });

export default VideoRendering;
