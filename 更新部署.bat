@echo off
REM 快速更新并推送到GitHub
chcp 65001 >nul

echo ========================================
echo  更新应用到Streamlit Cloud
echo ========================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

echo [1/4] 检查修改的文件...
git status
echo.

echo [2/4] 添加所有修改...
git add .
echo.

echo [3/4] 输入提交信息：
set /p commit_msg="请输入提交信息: "
if "%commit_msg%"=="" set commit_msg=更新代码

git commit -m "%commit_msg%"
echo.

echo [4/4] 推送到GitHub...
git push origin main
echo.

if errorlevel 1 (
    echo ❌ 推送失败！请检查错误信息
    pause
    exit /b 1
)

echo ========================================
echo ✅ 更新成功！
echo.
echo Streamlit Cloud正在自动部署...
echo 预计2-5分钟后生效
echo.
echo 你可以访问应用查看更新进度
echo ========================================
echo.

pause

