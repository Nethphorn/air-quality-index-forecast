import prettier from 'eslint-config-prettier'
import path from 'node:path'
import { includeIgnoreFile } from '@eslint/compat'
import js from '@eslint/js'
import noCommentedCode from 'eslint-plugin-no-commented-code'
import { defineConfig } from 'eslint/config'
import globals from 'globals'
import ts from 'typescript-eslint'

const gitignore_path = path.resolve(import.meta.dirname, '.gitignore')

export default defineConfig(
  includeIgnoreFile(gitignore_path),
  {
    ignores: [
      '*.config.{ts,js,cjs,mjs}',
      'dist/**',
      'build/**',
      '.venv/**',
      'node_modules/**'
    ]
  },
  js.configs.recommended,
  ts.configs.recommended,
  prettier,
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    plugins: {
      'no-commented-code': noCommentedCode
    },
    rules: {
      'no-undef': 'off',
      'no-var': 'error',
      'prefer-const': 'error',
      'no-console': ['error', { allow: ['warn', 'error', 'info'] }],
      'no-debugger': 'error',
      complexity: ['error', 15],
      'max-depth': ['error', 2],
      'no-commented-code/no-commented-code': 'warn',
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Literal[value=null]',
          message: 'Use undefined instead of null.'
        }
      ],
      '@typescript-eslint/naming-convention': [
        {
          selector: 'variable',
          types: ['boolean'],
          format: ['snake_case'],
          prefix: ['is_', 'has_', 'can_', 'should_', 'will_', 'did_'],
          leadingUnderscore: 'allow',
          trailingUnderscore: 'allow'
        },
        {
          selector: 'variable',
          format: ['snake_case', 'UPPER_CASE'],
          leadingUnderscore: 'allow',
          trailingUnderscore: 'allow'
        },
        {
          selector: 'function',
          format: ['snake_case'],
          leadingUnderscore: 'allow',
          trailingUnderscore: 'allow'
        }
      ]
    }
  },
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.mts', '**/*.cts'],
    languageOptions: {
      parser: ts.parser,
      parserOptions: {
        projectService: true
      }
    }
  }
)
