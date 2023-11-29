import React from 'react';
import { View, Text, TouchableOpacity, ImageBackground, StyleSheet } from 'react-native';
import Icon from 'react-native-vector-icons/FontAwesome';

const ModulesScreen = ({ navigation }) => {
  const sections = [
    "Motion Detection",
    "Dataset Collection For Face",
    "Face Video Recognition using FaceNet CNN",
    "Car Image Detection using CNN",
    "Car Video Detection using Yolov5",
  ];

  return (
    <ImageBackground
      source={require('../assets/mainscreen.png')} // Replace with your background image
      style={styles.container}
    >
      <View style={styles.overlay}>
        <Text style={styles.title}>MODULES</Text>
        {sections.map((section, index) => (
          <TouchableOpacity
            key={index}
            style={styles.sectionBox}
            onPress={() => {
              if (section === "Car Image Detection using CNN") {
                navigation.navigate('CarDetection');
              } else if (section === "Dataset Collection For Face") {
                navigation.navigate('FaceRecognition');
              } else if (section === "Car Video Detection using Yolov5") {
                navigation.navigate('YoloDetection');
              } else if (section === "Face Video Recognition using FaceNet CNN") {
                navigation.navigate('FaceDetection');
              }
              else {
                console.log(`Pressed: ${section}`);
              }
            }}
          >
            <Text style={styles.sectionText}>{section}</Text>
          </TouchableOpacity>
        ))}
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
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlay: {
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    flex: 1,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    color: 'white',
    marginBottom: 40,
    fontWeight: 'bold',
    fontSize: 40,
  },
  sectionBox: {
    width: '80%',
    height: 80,
    backgroundColor: '#000000',
    borderRadius: 10,
    marginBottom: 20,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
    padding:15,
    shadowColor: '#FFFFFF',
    shadowOffset: { width: 10, height: 2 },
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 2,
  },
  sectionText: {
    fontSize: 18,
    fontWeight: 'bold',
    textAlign:'center',
    color: 'white',
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
});

export default ModulesScreen;

