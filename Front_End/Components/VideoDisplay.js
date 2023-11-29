// VideoDisplay.js
import React, { useState } from 'react';
import { View, TouchableOpacity, Text } from 'react-native';
import { Video } from 'expo-av';
import Icon from 'react-native-vector-icons/FontAwesome';

const VideoDisplay = ({ videoUri }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = React.createRef();

  const togglePlayPause = async () => {
    if (videoRef.current) {
      if (isPlaying) {
        await videoRef.current.pauseAsync();
      } else {
        await videoRef.current.playAsync();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <View style={{marginTop:-50, backgroundColor:'rgba(0,0,0,0.4)'}}>
      <Video ref={videoRef} source={{ uri: videoUri }} shouldPlay={true} resizeMode="contain" style={{ width: 300, height: 300, borderRadius:5, }} />
      <TouchableOpacity onPress={togglePlayPause}>
        <Text style={{marginVertical:10, textAlign:'center'}}>{isPlaying ? <Icon name="pause-circle-o" size={40} color="#FFFFFF" /> : <Icon name="play-circle-o" size={40} color="#FFFFFF" />}</Text>
      </TouchableOpacity>
    </View>
  );
};

export default VideoDisplay;
