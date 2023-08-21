1.  give the permission to the script files
     ```
     sudo chmod 777 -R ./*.sh
     ```

2. setup the env and install tritonserver
      ```
      sudo ./triton_setup.sh
      ```

3. setup the env of application 
      ```
      ./ app_setup.sh
      ```

4. give the permission to the script files
      cd tritonserver
      ```
      sudo chmod 777 -R ./*.sh
      ```

5. start the tritonserver
      ```
      ./start_server.sh
      ```

6. run the application
      ```
      ./run_demo.sh
      ```