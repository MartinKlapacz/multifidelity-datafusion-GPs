#!/usr/bin/bash

pyreverse ./src && 
dot -Tpng ./classes.dot -o UML_export.png && 
mv UML_export.png ./figures &&
rm classes.dot packages.dot