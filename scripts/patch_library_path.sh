if [ $LD_LIBRARY_PATH ]
then
  HAS_PATH=`echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -c "${PLUGIN_PATH}"`
  # echo "HAS_PATH=${HAS_PATH}"
  if [ "${HAS_PATH}" == "0" ]
  then
    echo "Append ${PLUGIN_PATH} to LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH=${PLUGIN_PATH}:${LD_LIBRARY_PATH}
  fi
else
  echo "Set LD_LIBRARY_PATH to ${PLUGIN_PATH}"
  export LD_LIBRARY_PATH=${PLUGIN_PATH}
fi

