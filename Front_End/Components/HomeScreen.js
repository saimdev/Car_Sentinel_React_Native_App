import {React, useEffect, useState} from 'react';
import { View, Text, TouchableOpacity, ImageBackground, StyleSheet, Linking} from 'react-native';
import Icon from 'react-native-vector-icons/FontAwesome';

const HomeScreen = ({ navigation }) => {

  const [pdfLink, setPdfLink] = useState('https://drive.google.com/file/d/1Ja7A7hH0BNXlSAddvZabrWkhpcBBY3VU/view?usp=drive_link');

  const sections = [
    "Motion Detection",
    "Reports",
  ];

  return (
    <ImageBackground
      source={require('../assets/mainscreen.png')}
      style={styles.container}
    >
      <View style={styles.overlay}>
        <Text style={styles.title}>MENU</Text>
        {sections.map((section, index) => (
          <TouchableOpacity
            key={index}
            style={styles.sectionBox}
            onPress={() => {
              if (section === "Motion Detection") {
                navigation.navigate('MotionDetection')
              } else if (section=="Modules"){
                navigation.navigate('ModulesScreen')
              }else {
                Linking.openURL(pdfLink);
              }
            }}
          >
            <Text style={styles.sectionText}>{section}</Text>
          </TouchableOpacity>
        ))}
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

export default HomeScreen;

