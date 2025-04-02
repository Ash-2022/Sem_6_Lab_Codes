for file in *.txt; do
    case $file in
        test*.txt)
            echo "Test file: $file"
            ;;
        *.log.txt|*.txt.log)
            echo "Log file: $file"
            ;;
        *)
            echo "Other file: $file"
            ;;
    esac
done

for i in 1 2 3; do
    echo "Number $i"
done
