#!/bin/sh

set -e  # Выход при любой ошибке

DOCKER_IMAGE="2109199812/docker-latex"

pull_or_build_docker_image() {
    if [ "$1" = "build" ]; then  # Используем = вместо == для совместимости с POSIX sh
        echo "Сборка Docker-образа для LaTeX..."
        if ! command -v docker >/dev/null 2>&1; then
            echo "Ошибка: Docker не установлен или не найден в PATH" >&2
            exit 1
        fi
        docker build --load -t "$DOCKER_IMAGE" .
    else
        echo "Загрузка Docker-образа для LaTeX..."
        if ! command -v docker >/dev/null 2>&1; then
            echo "Ошибка: Docker не установлен или не найден в PATH" >&2
            exit 1
        fi
        docker pull "$DOCKER_IMAGE"
    fi
    
    echo "Готово! Образ $DOCKER_IMAGE доступен."
}

show_help() {
    echo "Скрипт для управления Docker-образом LaTeX"
    echo "Использование: $0 [build|pull|help]"
    echo ""
    echo "Команды:"
    echo "  build   - Собрать образ локально"
    echo "  pull    - Загрузить образ из Docker Hub (по умолчанию)"
    echo "  help    - Показать эту справку"
}

main() {
    case "${1:-pull}" in
        "build"|"pull")
            pull_or_build_docker_image "$1"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo "Неизвестная команда: $1" >&2
            show_help
            exit 1
            ;;
    esac
}

main "$@"