import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const Settings = () => {
  const [name, setName] = useState('');
  const [ipAddress, setIpAddress] = useState('');

  const handleSubmit = async () => {
    try {
        const userData = {
          name,
          ipAddress,
        };
        
        const userDataJSON = JSON.stringify(userData);
        
        await AsyncStorage.setItem('userData', userDataJSON);
        console.log("Saved", name, ipAddress);
        // navigation.replace('Home');
      } catch (error) {
        console.error('Error saving settings:', error);
      }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Enter Your Settings</Text>
      <TextInput
        style={styles.input}
        placeholder="Name"
        value={name}
        onChangeText={setName}
      />
      <TextInput
        style={styles.input}
        placeholder="IP Address"
        value={ipAddress}
        onChangeText={setIpAddress}
      />
      <Button title="Save" onPress={handleSubmit} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    marginBottom: 20,
  },
  input: {
    width: '80%',
    padding: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#ccc',
  },
});

export default Settings;
