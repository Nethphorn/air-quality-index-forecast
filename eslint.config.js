// ESLint flat configuration file
export default [
    {
        ignores: ["node_modules", ".venv", "dist", "build"]
    },
    {
        files: ["**/*.js"],
        languageOptions: {
            ecmaVersion: "latest",
            sourceType: "module"
        },
        rules: {
            "no-unused-vars": "warn",
            "no-console": "off"
        }
    }
];
